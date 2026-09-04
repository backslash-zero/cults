<script lang="ts">
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
</script>

<h1 class="text-2xl font-terminal-grotesque">MIVILUDES</h1>

<p class="text-gray-700 dark:text-gray-300">
	The third full-text corpus, alongside Interviews and Literature: MIVILUDES's own publications,
	chunked, annotated, and embedded the same way. One of its two source documents — "Comment
	identifier une dérive sectaire ?" — is also the citation source for every entry in
	<a href="/cult-spaces/miviludes-criteria" class="hover:underline">MIVILUDES criteria</a>: that page
	distills the same webpage into discrete criteria, this one treats it as continuous text.
</p>

<ul class="flex flex-col gap-2">
	{#each data.documents as doc (doc.documentId)}
		<li class="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
			<a href={doc.url} target="_blank" rel="noopener noreferrer" class="font-terminal-grotesque hover:underline">
				{doc.title}
			</a>
			<p class="text-gray-500 dark:text-gray-400 text-sm">
				{doc.itemsEmbedded.toLocaleString()} chunks embedded
			</p>
		</li>
	{/each}
</ul>
