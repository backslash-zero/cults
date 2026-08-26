<script lang="ts">
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	let { interview } = $derived(data);
</script>

<a href="/cult-spaces/interviews" class="text-gray-500 dark:text-gray-400 hover:underline"
	>&larr; Interviews</a
>

<h1 class="text-2xl font-terminal-grotesque mt-2">{interview.corpusId}</h1>

<dl class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-sm text-gray-600 dark:text-gray-400 my-4">
	<div><dt class="font-terminal-grotesque">Batch</dt><dd>{interview.batch}</dd></div>
	<div><dt class="font-terminal-grotesque">Date</dt><dd>{interview.dateTime}</dd></div>
	<div><dt class="font-terminal-grotesque">Method</dt><dd>{interview.method}</dd></div>
	<div><dt class="font-terminal-grotesque">Language</dt><dd>{interview.language}</dd></div>
	<div><dt class="font-terminal-grotesque">Location</dt><dd>{interview.location}</dd></div>
	<div><dt class="font-terminal-grotesque">Age</dt><dd>{interview.age}</dd></div>
	<div><dt class="font-terminal-grotesque">Gender</dt><dd>{interview.gender}</dd></div>
	<div><dt class="font-terminal-grotesque">Nationality</dt><dd>{interview.nationality}</dd></div>
	<div><dt class="font-terminal-grotesque">Word count</dt><dd>{interview.totalWordCount}</dd></div>
</dl>

<article class="prose dark:prose-invert max-w-none">
	{#each (interview.text ?? '').split('\n\n') as paragraph, i (i)}
		<p>{paragraph}</p>
	{/each}
</article>

{#if interview.translated && interview.translationText}
	<h2 class="text-xl font-terminal-grotesque mt-8">
		Translation ({interview.translationLanguage})
	</h2>
	<article class="prose dark:prose-invert max-w-none">
		{#each interview.translationText.split('\n\n') as paragraph, i (i)}
			<p>{paragraph}</p>
		{/each}
	</article>
{/if}
