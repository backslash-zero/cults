<script lang="ts">
	import EmptyState from '$lib/components/EmptyState.svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
</script>

<h1 class="text-2xl font-terminal-grotesque">Literature</h1>

{#if data.items.length === 0}
	<EmptyState message="Not started yet — the literature corpus hasn't been migrated into this pipeline." />
{:else}
	<ul class="flex flex-col gap-2">
		{#each data.items as item (item.corpusId)}
			<li class="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
				<p class="font-terminal-grotesque">{item.title}</p>
				<p class="text-gray-500 dark:text-gray-400 text-sm">
					{item.authors?.join(', ')} · {item.year} · {item.type}
				</p>
			</li>
		{/each}
	</ul>
{/if}
