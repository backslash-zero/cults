<script lang="ts">
	import StatCard from '$lib/components/StatCard.svelte';
	import CorpusSummaryCard from '$lib/components/CorpusSummaryCard.svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	let { overview, stats, miviludesDocumentCount } = $derived(data);
</script>

<p class="text-2xl text-gray-700 dark:text-gray-300 font-terminal-grotesque prose">
	This project investigates the criteria through which particular objects, groups, and practices
	come to be considered "cults," comparing the semantic structures that emerge from scholarly
	literature, interviews with participants, and institutional material on sectarian drift.
</p>

<section class="flex flex-col gap-3">
	<h2 class="text-sm uppercase tracking-wide text-gray-500 dark:text-gray-400 font-terminal-grotesque">
		Corpora
	</h2>
	<div class="grid grid-cols-2 sm:grid-cols-3 gap-4">
		<CorpusSummaryCard title="Interviews" count={overview.interviewCount} href="/cult-spaces/interviews" />
		<CorpusSummaryCard title="Literature" count={overview.literatureCount} href="/cult-spaces/literature" />
		<CorpusSummaryCard title="MIVILUDES" count={miviludesDocumentCount} href="/cult-spaces/miviludes" />
	</div>
</section>

<section class="flex flex-col gap-3">
	<h2 class="text-sm uppercase tracking-wide text-gray-500 dark:text-gray-400 font-terminal-grotesque">
		Reference lists
	</h2>
	<div class="grid grid-cols-2 sm:grid-cols-3 gap-4">
		<CorpusSummaryCard
			title="MIVILUDES criteria"
			count={overview.miviludesCriteriaCount}
			href="/cult-spaces/miviludes-criteria"
		/>
		<CorpusSummaryCard
			title="Concept backbone"
			count={stats.countsBySourceDataset.concept_backbone ?? 0}
			href="/cult-spaces/concept-backbone"
		/>
	</div>
</section>

{#if overview.interviewCount > 0}
	<section class="flex flex-col gap-4">
		<h2 class="text-xl font-terminal-grotesque">Interview corpus stats</h2>
		<div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
			<StatCard value={overview.interviewCount} label="Interviews" />
			<StatCard value={overview.interviewStats.totalWordCount} label="Total words" />
			<StatCard value={overview.interviewStats.avgWordCount} label="Avg words / interview" />
			<StatCard value={overview.interviewStats.translatedCount} label="Translated" />
		</div>

		<div class="grid sm:grid-cols-3 gap-6 text-gray-700 dark:text-gray-300">
			<div>
				<h3 class="font-terminal-grotesque mb-1">By batch</h3>
				<ul>
					{#each Object.entries(overview.interviewStats.byBatch) as [batch, count] (batch)}
						<li>{batch}: {count}</li>
					{/each}
				</ul>
			</div>
			<div>
				<h3 class="font-terminal-grotesque mb-1">By language</h3>
				<ul>
					{#each Object.entries(overview.interviewStats.byLanguage) as [language, count] (language)}
						<li>{language}: {count}</li>
					{/each}
				</ul>
			</div>
			<div>
				<h3 class="font-terminal-grotesque mb-1">By method</h3>
				<ul>
					{#each Object.entries(overview.interviewStats.byMethod) as [method, count] (method)}
						<li>{method}: {count}</li>
					{/each}
				</ul>
			</div>
		</div>
	</section>
{/if}

<section class="flex flex-col gap-4">
	<h2 class="text-xl font-terminal-grotesque">Full-text extraction &amp; embedding pipeline</h2>
	<p class="text-gray-700 dark:text-gray-300">
		Separate from the bibliographic corpus above: {stats.registry.totalDocuments} source documents have
		been run through extraction, annotation, and embedding, producing {stats.totalPoints.toLocaleString()}
		embedded chunks and concepts, pooled into one {stats.sharedSpace.chosenDimensions}-d shared space
		capturing {(stats.sharedSpace.varianceAtK * 100).toFixed(1)}% of variance.
	</p>
	<div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
		<StatCard value={stats.registry.totalDocuments} label="Documents processed" />
		<StatCard value={stats.totalPoints.toLocaleString()} label="Embedded points" />
		<StatCard value={(stats.countsBySourceDataset.concept_backbone ?? 0).toLocaleString()} label="Concept backbone" />
		<StatCard value={`${stats.sharedSpace.chosenDimensions}-d`} label="Shared space (95% variance)" />
	</div>
	<div class="grid sm:grid-cols-3 gap-6 text-gray-700 dark:text-gray-300">
		<div>
			<h3 class="font-terminal-grotesque mb-1">Chunks embedded by corpus</h3>
			<ul>
				{#each Object.entries(stats.registry.byCorpus) as [corpus, info] (corpus)}
					<li>{corpus}: {info.itemsEmbedded.toLocaleString()} ({info.documents} docs)</li>
				{/each}
			</ul>
		</div>
	</div>
	<a href="/cult-spaces/explore" class="font-terminal-grotesque hover:underline">
		Explore the shared embedding space ({stats.totalPoints.toLocaleString()} points) →
	</a>
</section>
