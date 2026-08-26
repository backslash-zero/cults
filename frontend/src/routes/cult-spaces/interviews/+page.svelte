<script lang="ts">
	import EmptyState from '$lib/components/EmptyState.svelte';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
</script>

<h1 class="text-2xl font-terminal-grotesque">Interviews</h1>

{#if data.interviews.length === 0}
	<EmptyState message="Not started yet — no interviews synced." />
{:else}
	<ul class="flex flex-col gap-2">
		{#each data.interviews as interview (interview.corpusId)}
			<li>
				<a
					href={`/cult-spaces/interviews/${interview.corpusId}`}
					class="flex flex-wrap justify-between gap-2 rounded-lg border border-gray-200 dark:border-gray-700 p-3 hover:border-gray-400 dark:hover:border-gray-500"
				>
					<span class="font-terminal-grotesque">{interview.corpusId}</span>
					<span class="text-gray-500 dark:text-gray-400 text-sm">
						{interview.batch} · {interview.language} · {interview.method}
						{#if interview.translated}· translated{/if}
						· {interview.totalWordCount} words
					</span>
				</a>
			</li>
		{/each}
	</ul>
{/if}
