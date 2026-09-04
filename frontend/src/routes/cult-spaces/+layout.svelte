<script lang="ts">
	import { page } from '$app/state';
	import type { Snippet } from 'svelte';
	let { children }: { children: Snippet } = $props();

	const links = [
		{ href: '/cult-spaces', label: 'Overview' },
		{ href: '/cult-spaces/interviews', label: 'Interviews' },
		{ href: '/cult-spaces/literature', label: 'Literature' },
		{ href: '/cult-spaces/dictionaries', label: 'Dictionaries' },
		{ href: '/cult-spaces/custom-terms', label: 'Custom terms' },
		{ href: '/cult-spaces/miviludes-criteria', label: 'MIVILUDES criteria' },
		{ href: '/cult-spaces/explore', label: 'Explore' }
	];
</script>

<div class="max-w-3xl mx-auto flex flex-col gap-8">
	<nav class="flex flex-wrap gap-4 font-terminal-grotesque">
		{#each links as link (link.href)}
			<a
				href={link.href}
				class="text-gray-700 dark:text-gray-300 hover:underline"
				class:underline={page.url.pathname === link.href}
			>
				{link.label}
			</a>
		{/each}
	</nav>

	{@render children()}
</div>
