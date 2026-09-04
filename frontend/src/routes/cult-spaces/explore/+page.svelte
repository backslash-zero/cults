<script lang="ts">
	import EmbeddingExplorer from '$lib/components/EmbeddingExplorer.svelte';
	import EmbeddingLegend from '$lib/components/EmbeddingLegend.svelte';
	import EmbeddingDetailPanel from '$lib/components/EmbeddingDetailPanel.svelte';
	import { ALL_SOURCE_DATASETS, DEFAULT_VISIBLE_DATASETS, prefersDarkMode } from '$lib/pointColors';
	import type { PointMeta, ProjectionMethod, SourceDataset } from '$lib/types';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	const METHOD_LABELS: Record<ProjectionMethod, string> = { pca: 'PCA', umap: 'UMAP', tsne: 't-SNE' };

	let loading = $state(true);
	let points: PointMeta[] = $state([]);
	let positionsByMethod: Partial<Record<ProjectionMethod, [number, number, number][]>> = $state({});
	let method: ProjectionMethod = $state('umap');

	let visible: Record<SourceDataset, boolean> = $state(
		Object.fromEntries(
			ALL_SOURCE_DATASETS.map((d) => [d, DEFAULT_VISIBLE_DATASETS.includes(d)])
		) as Record<SourceDataset, boolean>
	);

	let searchInput = $state('');
	let searchTerm = $state('');
	let debounceHandle: ReturnType<typeof setTimeout>;
	$effect(() => {
		const value = searchInput;
		clearTimeout(debounceHandle);
		debounceHandle = setTimeout(() => (searchTerm = value), 200);
	});

	let selectedPoint: PointMeta | null = $state(null);
	let dark = $state(false);

	$effect(() => {
		dark = prefersDarkMode();
	});

	async function loadMethod(m: ProjectionMethod) {
		if (positionsByMethod[m]) return;
		const res = await fetch(`/data/positions-${m}.json`);
		positionsByMethod = { ...positionsByMethod, [m]: await res.json() };
	}

	$effect(() => {
		(async () => {
			const metaRes = await fetch('/data/points-meta.json');
			points = await metaRes.json();
			await loadMethod(method);
			loading = false;
		})();
	});

	async function selectMethod(m: ProjectionMethod) {
		method = m;
		if (!positionsByMethod[m]) {
			loading = true;
			await loadMethod(m);
			loading = false;
		}
	}

	function toggleDataset(dataset: SourceDataset) {
		visible = { ...visible, [dataset]: !visible[dataset] };
	}
	function showAll() {
		visible = Object.fromEntries(ALL_SOURCE_DATASETS.map((d) => [d, true])) as Record<SourceDataset, boolean>;
	}
	function hideAll() {
		visible = Object.fromEntries(ALL_SOURCE_DATASETS.map((d) => [d, false])) as Record<SourceDataset, boolean>;
	}
</script>

<div class="flex flex-col gap-4">
	<div>
		<h1 class="text-xl font-terminal-grotesque">Explore the shared embedding space</h1>
		<p class="text-gray-700 dark:text-gray-300 text-sm mt-1">
			{data.stats.totalPoints.toLocaleString()} points — literature and MIVILUDES excerpts, interview
			chunks, MIVILUDES criteria, and the concept backbone — pooled into one {data.stats.sharedSpace
				.chosenDimensions}-d space ({(data.stats.sharedSpace.varianceAtK * 100).toFixed(1)}% variance),
			projected here to 3-D by PCA, UMAP, or t-SNE.
		</p>
	</div>

	<div class="flex flex-wrap items-center gap-4">
		<div class="flex gap-2 font-terminal-grotesque">
			{#each Object.entries(METHOD_LABELS) as [m, label] (m)}
				<button
					type="button"
					class="px-3 py-1 rounded-lg border border-gray-200 dark:border-gray-700"
					class:underline={method === m}
					onclick={() => selectMethod(m as ProjectionMethod)}
				>
					{label}
				</button>
			{/each}
		</div>
		<input
			type="search"
			placeholder="Search labels…"
			bind:value={searchInput}
			class="flex-1 min-w-48 rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent px-3 py-1 text-sm"
		/>
	</div>
</div>

<div class="mx-[calc(50%-50vw)] w-screen mt-4">
	<div class="max-w-7xl mx-auto grid md:grid-cols-[1fr_16rem] gap-4 px-4">
		<div class="relative h-[70vh] rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
			{#if loading}
				<div class="absolute inset-0 flex items-center justify-center">
					<p class="text-gray-500 dark:text-gray-400">Loading 43,415 points…</p>
				</div>
			{:else}
				<EmbeddingExplorer
					{points}
					positions={positionsByMethod[method] ?? []}
					{visible}
					{searchTerm}
					onSelect={(p) => (selectedPoint = p)}
				/>
			{/if}
		</div>
		<div class="flex flex-col gap-4">
			<EmbeddingLegend
				{visible}
				counts={data.stats.countsBySourceDataset}
				{dark}
				onToggle={toggleDataset}
				onShowAll={showAll}
				onHideAll={hideAll}
			/>
			<EmbeddingDetailPanel point={selectedPoint} />
		</div>
	</div>
</div>
