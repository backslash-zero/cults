<script lang="ts">
	import type { SourceDataset } from '$lib/types';
	import { ALL_SOURCE_DATASETS, SOURCE_DATASET_LABELS, colorForDataset } from '$lib/pointColors';

	let {
		visible,
		counts,
		dark,
		onToggle,
		onShowAll,
		onHideAll
	}: {
		visible: Record<SourceDataset, boolean>;
		counts: Record<string, number>;
		dark: boolean;
		onToggle: (dataset: SourceDataset) => void;
		onShowAll: () => void;
		onHideAll: () => void;
	} = $props();
</script>

<div class="flex flex-col gap-2">
	<div class="flex items-center justify-between">
		<h2 class="text-sm uppercase tracking-wide text-gray-500 dark:text-gray-400 font-terminal-grotesque">
			Categories
		</h2>
		<div class="flex gap-3 text-xs">
			<button type="button" class="text-gray-500 dark:text-gray-400 hover:underline" onclick={onShowAll}>
				Show all
			</button>
			<button type="button" class="text-gray-500 dark:text-gray-400 hover:underline" onclick={onHideAll}>
				Hide all
			</button>
		</div>
	</div>
	<ul class="flex flex-col gap-1">
		{#each ALL_SOURCE_DATASETS as dataset (dataset)}
			<li>
				<label class="flex items-center gap-2 cursor-pointer text-sm">
					<input type="checkbox" checked={visible[dataset]} onchange={() => onToggle(dataset)} />
					<span
						class="inline-block w-3 h-3 rounded-full shrink-0"
						style={`background-color: ${colorForDataset(dataset, dark)}`}
					></span>
					<span class="text-gray-700 dark:text-gray-300">{SOURCE_DATASET_LABELS[dataset]}</span>
					<span class="text-gray-400 dark:text-gray-500 ml-auto tabular-nums">
						{(counts[dataset] ?? 0).toLocaleString()}
					</span>
				</label>
			</li>
		{/each}
	</ul>
</div>
