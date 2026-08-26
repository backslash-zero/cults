<script lang="ts">
	let { value, label }: { value: string | number; label: string } = $props();
</script>

<div class="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
	<p class="text-3xl font-terminal-grotesque">{value}</p>
	<p class="text-sm text-gray-500 dark:text-gray-400">{label}</p>
</div>
