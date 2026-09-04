<script lang="ts">
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	let { item } = $derived(data);
</script>

<a href="/cult-spaces/literature" class="text-gray-500 dark:text-gray-400 hover:underline"
	>&larr; Literature</a
>

<h1 class="text-2xl font-terminal-grotesque mt-2">{item.title}</h1>

<dl class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-sm text-gray-600 dark:text-gray-400 my-4">
	<div><dt class="font-terminal-grotesque">Authors</dt><dd>{item.authors?.join(', ')}</dd></div>
	<div><dt class="font-terminal-grotesque">Year</dt><dd>{item.year}</dd></div>
	<div><dt class="font-terminal-grotesque">Type</dt><dd>{item.type}</dd></div>
	<div><dt class="font-terminal-grotesque">Source</dt><dd>{item.source}</dd></div>
	{#if item.doi}
		<div>
			<dt class="font-terminal-grotesque">DOI</dt>
			<dd><a href={`https://doi.org/${item.doi}`} class="hover:underline">{item.doi}</a></dd>
		</div>
	{/if}
	{#if item.isbn}
		<div><dt class="font-terminal-grotesque">ISBN</dt><dd>{item.isbn}</dd></div>
	{/if}
	<div><dt class="font-terminal-grotesque">Language</dt><dd>{item.language}</dd></div>
	{#if item.tags && item.tags.length > 0}
		<div><dt class="font-terminal-grotesque">Tags</dt><dd>{item.tags.join(', ')}</dd></div>
	{/if}
	{#if item.dateAdded}
		<div><dt class="font-terminal-grotesque">Date added</dt><dd>{item.dateAdded}</dd></div>
	{/if}
</dl>

{#if item.abstract}
	<h2 class="text-xl font-terminal-grotesque mt-8">Abstract</h2>
	<article class="prose dark:prose-invert max-w-none">
		<p>{item.abstract}</p>
	</article>
{/if}

{#if item.notes}
	<h2 class="text-xl font-terminal-grotesque mt-8">Notes</h2>
	<article class="prose dark:prose-invert max-w-none">
		<p>{item.notes}</p>
	</article>
{/if}
