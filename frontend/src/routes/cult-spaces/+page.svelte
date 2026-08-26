<script lang="ts">
	import StatCard from '$lib/components/StatCard.svelte';
	import CorpusSummaryCard from '$lib/components/CorpusSummaryCard.svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	let { overview } = $derived(data);
</script>

<p class="text-2xl text-gray-700 dark:text-gray-300 font-terminal-grotesque prose">
	My research investigates how unstable, contested concepts can be formally defined and
	represented as semantic spaces—using "cult" as a primary case study. The project originated from
	asking whether AI psychosis could be understood as a form of sectarian drift.
</p>

<section class="grid grid-cols-2 sm:grid-cols-4 gap-4">
	<CorpusSummaryCard title="Interviews" count={overview.interviewCount} href="/cult-spaces/interviews" />
	<CorpusSummaryCard title="Literature" count={overview.literatureCount} href="/cult-spaces/literature" />
	<CorpusSummaryCard
		title="Dictionaries"
		count={overview.dictionaryCount}
		href="/cult-spaces/dictionaries"
	/>
	<CorpusSummaryCard
		title="Custom terms"
		count={overview.customTermCount}
		href="/cult-spaces/custom-terms"
	/>
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
