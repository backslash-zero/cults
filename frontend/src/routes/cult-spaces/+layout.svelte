<script lang="ts">
	import { page } from '$app/state';
	import type { Snippet } from 'svelte';
	let { children }: { children: Snippet } = $props();

	const overviewLink = { href: '/cult-spaces', label: 'Overview' };
	// Corpora: full-text collections that get chunked, annotated, and embedded.
	const corpusLinks = [
		{ href: '/cult-spaces/interviews', label: 'Interviews' },
		{ href: '/cult-spaces/literature', label: 'Literature' },
		{ href: '/cult-spaces/miviludes', label: 'MIVILUDES' }
	];
	// Reference lists: curated, not extracted -- embedded as anchor points that
	// help structure the shared space, per thesis/03_Content/3_Methods.tex:184.
	const referenceLinks = [
		{ href: '/cult-spaces/miviludes-criteria', label: 'MIVILUDES criteria' },
		{ href: '/cult-spaces/concept-backbone', label: 'Concept backbone' }
	];
	const exploreLink = { href: '/cult-spaces/explore', label: 'Explore' };
</script>

{#snippet navLink(link: { href: string; label: string })}
	<a
		href={link.href}
		class="text-gray-700 dark:text-gray-300 hover:underline"
		class:underline={page.url.pathname === link.href}
	>
		{link.label}
	</a>
{/snippet}

<div class="max-w-3xl mx-auto flex flex-col gap-8">
	<nav class="flex flex-wrap items-baseline gap-x-6 gap-y-2 font-terminal-grotesque">
		{@render navLink(overviewLink)}
		<span class="flex flex-wrap gap-4">
			{#each corpusLinks as link (link.href)}
				{@render navLink(link)}
			{/each}
		</span>
		<span class="flex flex-wrap gap-4">
			{#each referenceLinks as link (link.href)}
				{@render navLink(link)}
			{/each}
		</span>
		{@render navLink(exploreLink)}
	</nav>

	{@render children()}
</div>
