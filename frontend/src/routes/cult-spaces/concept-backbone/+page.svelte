<script lang="ts">
	import StatCard from '$lib/components/StatCard.svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	let { stats } = $derived(data);
</script>

<h1 class="text-2xl font-terminal-grotesque">Concept backbone</h1>

<p class="text-gray-700 dark:text-gray-300">
	A generic, low-bias list of abstract English concepts drawn from WordNet, selected without
	reference to cults or sects, that serves as neutral anchor points structuring the shared
	embedding space — it actively participates in fitting that space, rather than being compared
	against it after the fact.
</p>

<div class="grid grid-cols-2 sm:grid-cols-3 gap-4">
	<StatCard value={(stats.countsBySourceDataset.concept_backbone ?? 0).toLocaleString()} label="Concepts" />
</div>

<a href="/cult-spaces/explore?dataset=concept_backbone" class="font-terminal-grotesque hover:underline">
	Explore the concept backbone in 3-D →
</a>
