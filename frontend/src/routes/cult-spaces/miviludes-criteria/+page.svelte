<script lang="ts">
	import EmptyState from '$lib/components/EmptyState.svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
</script>

<h1 class="text-2xl font-terminal-grotesque">MIVILUDES criteria</h1>

<p class="text-gray-700 dark:text-gray-300">
	The Miviludes' sect-identification criteria, bilingually (FR/EN), used as reference points in the
	shared embedding space.
</p>

{#if data.criteria.length === 0}
	<EmptyState message="No criteria synced yet — run the Sanity sync." />
{:else}
	<ol class="flex flex-col gap-3">
		{#each data.criteria as criterion (criterion.corpusId)}
			<li class="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
				<p class="text-gray-800 dark:text-gray-200">{criterion.order}. {criterion.criterionFr}</p>
				<p class="text-gray-500 dark:text-gray-400 mt-1">{criterion.criterionEn}</p>
				{#if criterion.citation}
					<p class="text-xs text-gray-400 dark:text-gray-500 mt-2">{criterion.citation}</p>
				{/if}
			</li>
		{/each}
	</ol>
{/if}
