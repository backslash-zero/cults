<script lang="ts">
	let { title, count, href }: { title: string; count: number; href: string } = $props();
</script>

<a
	{href}
	class="block rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:border-gray-400 dark:hover:border-gray-500"
>
	<p class="text-xl font-terminal-grotesque">{title}</p>
	{#if count > 0}
		<p class="text-gray-500 dark:text-gray-400">{count} {count === 1 ? 'entry' : 'entries'}</p>
	{:else}
		<p class="text-gray-400 dark:text-gray-500 italic">Not started yet</p>
	{/if}
</a>
