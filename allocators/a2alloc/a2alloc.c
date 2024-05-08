#undef NDEBUG
#define _GNU_SOURCE

#include <stdlib.h>
#include <pthread.h>
#include "memlib.h"
#include "mm_thread.h"

/* Constants */
#define PAGE_SIZE 4096	   // Hardcoded for x86_64, page size is always 4KB.
#define CACHELINE_SIZE 64  // Hardcoded from /proc/cpuinfo.
#define NUM_CLASS 9		   // Number of block class sizes supported.
#define BASE_CLASS 3 // Smallest block class to the power of 2

/* Structs */

/**
 * @brief Represents an entry in the big freelist.
 * Pointers to a big_freelist struct point to the region of memory represented
 * by each big_freelist entry, which is of size (num_pages * pagesize).
 */
struct big_freelist
{
	int num_pages;
	struct big_freelist *next;
};

/**
 * @brief Represents a free list entry for a given pageref.
 */
struct freelist
{
	struct freelist *next;
};

/**
 * @brief Represents a page of memory that is ready to be allocated.
 * The actual memory itself is in freelist,
 * and is 4096 bytes starting at the address freelist_base.
 */
struct pageref
{
	struct pageref *next;
	struct freelist *freelist;
	char *freelist_base;
	int num_free;
};

/* Global Variables */
pthread_mutex_t sbrk_lock = PTHREAD_MUTEX_INITIALIZER; /* lock for sbrk calls (heap access) */

static struct big_freelist *big_list = NULL;			   /* The head of our big freelist. NULL if the big freelist is empty. */
pthread_mutex_t big_list_lock = PTHREAD_MUTEX_INITIALIZER; /* lock for big freelist */

static struct pageref *new_free_page_refs = NULL; /* The head of our list of free references. NULL if the list is empty. */
static struct pageref *reusable_page_refs = NULL;
pthread_mutex_t free_refs_lock = PTHREAD_MUTEX_INITIALIZER; /* lock for free refs */

static char *processor_locks_base; /* pointer to base address for per-processor locks. */
static int num_processors;

/* Private Helper Functions */

/**
 * @brief Allocate sz memory where sz > (1/2) pagesize.
 * Allocates in terms of whole pages only, so max fragmentation is 1 page or
 * less. As per the assignment specification, large allocations are rare -
 * so we'll take the performance hit and serialize them.
 *
 * @param sz size of memory acquired
 */
static void *big_mm_malloc(size_t sz)
{
	void *result = NULL;

	int num_pages = (sz + PAGE_SIZE - 1) / PAGE_SIZE;
	pthread_mutex_lock(&big_list_lock);
	struct big_freelist *curr = big_list; /* pointer to current location in big freelist */
	struct big_freelist *prior = NULL;	  /* pointer to prior item in big freelist. Needed if we remove an entire entry from freelist. */

	/* Iterate through the big freelist to look for a large enough chunk of memory. */
	while (curr != NULL)
	{
		if (curr->num_pages > num_pages)
		{ /* larger entry than needed - cut into two and take the part we need. */
			/*
			 * We take the latter half of the entry as our memory,
			 * to minimize how much metadata we need to update.
			 */
			curr->num_pages -= num_pages;
			int *header_ptr = (int *)((char *)curr + (curr->num_pages * PAGE_SIZE));
			*header_ptr = num_pages;
			result = (void *)((char *)header_ptr + sizeof(int));
			break;
		}
		else if (curr->num_pages == num_pages)
		{ /* Perfectly sized entry - remove from list and use. */
			/* Update freelist. */
			if (prior != NULL)
				prior->next = curr->next;
			else
				big_list = curr->next;

			/* Assign pointer. */
			int *header_ptr = (int *)curr;
			result = (void *)((char *)header_ptr + sizeof(int));
			break;
		}
		else
		{ /* Check the next item in freelist */
			prior = curr;
			curr = curr->next;
		}
	}

	pthread_mutex_unlock(&big_list_lock);

	/* No suitable memory is found in the freelist, so we need to allocate more. */
	if (result == NULL)
	{
		pthread_mutex_lock(&sbrk_lock);
		int *header_ptr = (int *)mem_sbrk(num_pages * PAGE_SIZE); /* get num_pages of memory from heap */
		pthread_mutex_unlock(&sbrk_lock);

		if (header_ptr != NULL)
		{
			*header_ptr = -1; /* record that this is a big page */
			header_ptr++;
			*header_ptr = num_pages; /* record num_pages in second sizeof(int) bytes */
			header_ptr++;
			result = (void *)header_ptr; /* set malloc pointer to remainder of allocated memory, after num_pages. */
		}
	}
	return result;
}

/**
 * @brief Free ptr where size of memory of ptr is > (1/2) pagesize.
 * Does not restore contiguity.
 *
 * @param ptr pointer of memory to be freed
 */
static void big_mm_free(void *ptr)
{
	/* Get num_pages from the header of ptr */
	int *header_ptr = (int *)((char *)ptr - sizeof(int));
	struct big_freelist *newly_freed = (struct big_freelist *)header_ptr;
	/* Add our newly freed pages to the front of the freelist. */
	pthread_mutex_lock(&big_list_lock);
	newly_freed->next = big_list;
	big_list = newly_freed;
	pthread_mutex_unlock(&big_list_lock);
}

/**
 * @brief Returns the index of the corresponding block class.
 * Indices range from 0 to NUM_CLASS (assuming 4KB pages), where
 * 2^(BASE_CLASS + index) is the block class.
 *
 * @param sz block size
 * @return the index of the corresponding block class
 */
static inline int get_block_class_index(size_t sz)
{
	for (int i = 0; i < NUM_CLASS; i++)
	{
		if ((1 << (i + BASE_CLASS)) >= sz)
			return i;
	}
	return -1; // should never occur
}

/**
 * @brief Returns the index of the corresponding processor.
 *
 * @return the index of the corresponding processor
 */
static inline int get_processor_index()
{
	int cpu_id = sched_getcpu() % num_processors;
	return cpu_id;
}

/**
 * @brief Returns the mutex used to lock processor heap.
 *
 * @param processor_index the index of the processor
 */
static inline pthread_mutex_t *processor_mutex(int processor_index)
{
	return (pthread_mutex_t *)(processor_locks_base +
							   (processor_index * CACHELINE_SIZE));
}

/**
 * @brief Returns a pointer to the head of the pageref list specified.
 *
 * @param processor_index the index of the processor
 * @param block_class_index the index of the block class
 * @return a pointer to the head of the pageref list
 */
static inline struct pageref *pageref_head(int processor_index,
										   int block_class_index)
{
	return (struct pageref *)(*(
		(u_int64_t *)(dseg_lo +
					  (processor_index * sizeof(void *) * NUM_CLASS) +
					  (block_class_index * sizeof(void *)))));
}

/**
 * @brief Set the pointer to the head of the pageref list specified.
 *
 * @param processor_index the index of the processor
 * @param block_class_index the index of the block class
 * @param address address to set as the head of the pageref list
 */
static inline void set_pageref_head(int processor_index,
									int block_class_index,
									struct pageref *address)
{
	*((u_int64_t *)(dseg_lo +
					(processor_index * sizeof(void *) * NUM_CLASS) +
					(block_class_index * sizeof(void *)))) = (u_int64_t)address;
}

/**
 * @brief Returns a pointer to the head of a page given an address in the page.
 *
 * @param address an address in the page
 * @return a pointer to the page the address belongs to
 */
void *get_page_head(void *address)
{
	return dseg_lo + (((char *)address - dseg_lo) / PAGE_SIZE) * PAGE_SIZE;
}

/**
 * @brief Allocates a new page ref and puts it at the head.
 *
 * @param processor_index the index of the processor
 * @param block_class_index the index of the block class
 * @return a pointer to the pageref
 */
static struct pageref *allocate_pageref(int processor_index,
										int block_class_index)
{
	struct pageref *page_ref;
	pthread_mutex_lock(&free_refs_lock);

	int create_freelist = 0;

	if (reusable_page_refs != NULL)
	{
		/* re-use free reference */
		page_ref = reusable_page_refs;
		reusable_page_refs = reusable_page_refs->next;
	}
	else
	{
		create_freelist = 1;
		if (new_free_page_refs == NULL)
		{
			/* Acquire memory for storing page_ref */
			pthread_mutex_lock(&sbrk_lock);
			page_ref = (struct pageref *)mem_sbrk(PAGE_SIZE);
			pthread_mutex_unlock(&sbrk_lock);

			/* Fill the memory with free page_refs */
			struct pageref *tmp;
			for (int i = CACHELINE_SIZE; i < PAGE_SIZE; i += CACHELINE_SIZE)
			{
				tmp = (struct pageref *)(((char *)page_ref) + i);
				tmp->next = new_free_page_refs;
				new_free_page_refs = tmp;
			}
		}
		else
		{
			/* Take avalibale free page_ref. */
			page_ref = new_free_page_refs;
			/* Move free ref list pointer to the next. */
			new_free_page_refs = new_free_page_refs->next;
		}
	}

	pthread_mutex_unlock(&free_refs_lock);
	/* Add new page to the head of the page_ref list */
	struct pageref *old_head = pageref_head(processor_index, block_class_index);
	page_ref->next = old_head;
	set_pageref_head(processor_index, block_class_index, page_ref);

	char *freelist_base_address = page_ref->freelist_base;
	if (create_freelist)
	{
		/* Build freelist of the page if we just created it */
		pthread_mutex_lock(&sbrk_lock);
		/* Allocate a page */
		freelist_base_address = (char *)mem_sbrk(PAGE_SIZE);
		pthread_mutex_unlock(&sbrk_lock);
		page_ref->freelist_base = freelist_base_address;
	}

	page_ref->freelist = NULL;
	page_ref->num_free = 0;

	struct freelist *free_list_tmp;
	/* Create freelist entries from newly allocated page */
	for (int i = 0; i < PAGE_SIZE; i += (1 << (block_class_index + BASE_CLASS)))
	{
		free_list_tmp = (struct freelist *)(freelist_base_address + i);
		free_list_tmp->next = page_ref->freelist;
		page_ref->freelist = free_list_tmp;
		page_ref->num_free += 1;
	}

	/* Add metadata in the beginning of the page */
	*page_ref->freelist_base = processor_index;
	*(page_ref->freelist_base + sizeof(int)) = block_class_index;
	return page_ref;
}

/**
 * @brief Allocate sz memory for sz <= (1/2) pagesize.
 *
 * @param sz aquired memory size
 * @return pointer to the start of aquired memory
 */
static void *subpage_mm_malloc(size_t sz)
{
	int processor_index = get_processor_index();
	int block_class_index = get_block_class_index(sz);

	pthread_mutex_lock(processor_mutex(processor_index));
	struct pageref *page_ref = pageref_head(processor_index, block_class_index);

	/* Find first page ref with free space */
	while (page_ref != NULL)
	{
		if (page_ref->num_free > 0)
		{
			/*
			 * if it is the first block in the page,
			 * actual allocable space is 8 byte smaller.
			 */
			if ((void *)page_ref->freelist == (void *)page_ref->freelist_base)
			{
				if ((1 << (BASE_CLASS + block_class_index)) - 8 >= sz)
					break;
				else if (page_ref->num_free > 1)
				{
					/* swap next free block with this special block */
					struct freelist *old_freelist_next = page_ref->freelist->next;
					page_ref->freelist->next = old_freelist_next->next;
					old_freelist_next->next = page_ref->freelist;
					page_ref->freelist = old_freelist_next;
					break;
				}
			}
			else
				break;
		}
		page_ref = page_ref->next;
	}

	/* No avaliable page ref found, need to allocate new page ref. */
	if (page_ref == NULL)
		page_ref = allocate_pageref(processor_index, block_class_index);

	int *memory = (int *)page_ref->freelist;

	page_ref->num_free -= 1; // decrement freelist count
	page_ref->freelist = ((struct freelist *)memory)->next;

	if (memory == (void *)page_ref->freelist_base)
		memory = (int *)(((char *)memory) + 2 * sizeof(int));
	pthread_mutex_unlock(processor_mutex(processor_index));
	return memory;
}

/**
 * @brief Free ptr, where ptr size is <= (1/2) pagesize.
 *
 * @param ptr pointer of memory to be freed
 * @return 1 if successful and 0 on failure
 */
static int subpage_mm_free(void *ptr)
{
	void *page_head = get_page_head(ptr);
	int processor_index = *(((int *)page_head));
	if (processor_index == -1)
		return 0; // big malloc
	int block_class_index = *((int *)(page_head + sizeof(int)));

	pthread_mutex_lock(processor_mutex(processor_index));
	/* Get the page ref to return memory to */
	struct pageref *page_ref_head = pageref_head(processor_index, block_class_index);
	struct pageref *page_ref = page_ref_head;
	struct pageref *prior_page_ref = NULL;

	while (page_ref != NULL)
	{
		if (((void *)page_ref->freelist_base) == page_head)
			break;
		prior_page_ref = page_ref;
		page_ref = page_ref->next;
	}

	/* Add newly freed memory to the freelist */
	struct freelist *tmp = page_ref->freelist;
	page_ref->freelist = (struct freelist *)ptr;
	page_ref->freelist->next = tmp;

	page_ref->num_free += 1;
	/* If the page is fully empty, share the page for reuse */
	if (page_ref->num_free == (PAGE_SIZE / (1 << (BASE_CLASS + block_class_index))))
	{
		/* Remove page ref from heap */
		if (prior_page_ref != NULL)
			prior_page_ref->next = page_ref->next;
		else
			set_pageref_head(processor_index, block_class_index, page_ref->next);

		bzero(page_ref->freelist_base, PAGE_SIZE); // zero out freelist

		/* Update reusable refs */
		pthread_mutex_lock(&free_refs_lock);
		page_ref->next = reusable_page_refs;
		reusable_page_refs = page_ref; // make page_ref new head
		pthread_mutex_unlock(&free_refs_lock);
	}
	pthread_mutex_unlock(processor_mutex(processor_index));
	return 1;
}

/* Public Methods */

void *mm_malloc(size_t sz)
{
	void *result;
	if (sz <= (PAGE_SIZE / 2))
		result = subpage_mm_malloc(sz);
	else
		result = big_mm_malloc(sz + 2 * sizeof(int));
	return result;
}

void mm_free(void *ptr)
{
	if (ptr == NULL)
		return;
	if (!subpage_mm_free(ptr))
		big_mm_free(ptr);
}

int mm_init(void)
{
	/* Initialize heap */
	if (dseg_lo == NULL && dseg_hi == NULL)
	{
		if (mem_init() == -1)
			return -1;
	}

	num_processors = getNumProcessors();

	/* Allocate memory for free lists and per-processor mutexes */
	int freelist_space = NUM_CLASS * sizeof(void *) * num_processors;
	int mutex_space = CACHELINE_SIZE * num_processors;
	int page_needed = (freelist_space + mutex_space + PAGE_SIZE - 1) / PAGE_SIZE;
	char *ref = mem_sbrk(page_needed * PAGE_SIZE);
	bzero(ref, page_needed * PAGE_SIZE);

	/* Initialize mutexes */
	processor_locks_base = ref + freelist_space;
	for (int i = 0; i < num_processors; i++)
	{
		pthread_mutex_t *temp_lock = (pthread_mutex_t *)(processor_locks_base + (CACHELINE_SIZE * i));
		pthread_mutex_init(temp_lock, NULL);
	}
	return 0;
}
