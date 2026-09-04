<script lang="ts">
	import type { PointMeta } from '$lib/types';
	import { SOURCE_DATASET_LABELS } from '$lib/pointColors';

	let { point }: { point: PointMeta | null } = $props();
</script>

<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex flex-col gap-2 min-h-[8rem]">
	{#if point}
		<p class="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 font-terminal-grotesque">
			{SOURCE_DATASET_LABELS[point.source_dataset]}
		</p>
		<p class="text-gray-700 dark:text-gray-300 text-sm">{point.label}</p>
		{#if point.label_en && point.label_en !== point.label}
			<p class="text-gray-500 dark:text-gray-400 text-sm italic">{point.label_en}</p>
		{/if}
		{#if point.response_rank}
			<p class="text-xs text-gray-400 dark:text-gray-500">Response #{point.response_rank}</p>
		{/if}
		<p class="text-xs text-gray-400 dark:text-gray-500 font-mono break-all">{point.key}</p>
	{:else}
		<p class="text-gray-400 dark:text-gray-500 italic text-sm">Click a point to see its details.</p>
	{/if}
</div>
