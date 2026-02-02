<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { createEventDispatcher, getContext } from 'svelte';

	const i18n = getContext('i18n');
	const hoverHandler = getContext('reasoningTreeHover');
	const dispatch = createEventDispatcher();

	type $$Props = NodeProps;
	export let data: $$Props['data'];

	// Hover delay timer for quick highlight trigger
	let hoverTimer: ReturnType<typeof setTimeout> | null = null;
	const HOVER_DELAY = 300; // 300ms for responsive feel while avoiding flicker
	let isHovering = false; // Track hover state to prevent redundant dispatches

	const actionDisplay = (action?: string) => {
		if (!action) return $i18n.t('Step');
		return action;
	};

	$: highlightTarget =
		data?.segment_text ?? // Prefer the actual extracted segment from backend
		data?.highlightText ??
		data?.reasoning_excerpt ??
		data?.detail ??
		data?.content ??
		data?.title ??
		'';

	$: targetPosition = data?.layoutDirection === 'horizontal' ? Position.Left : Position.Top;
	$: sourcePosition = data?.layoutDirection === 'horizontal' ? Position.Right : Position.Bottom;
</script>

<div
	class={`px-3 py-2.5 rounded-2xl border bg-white dark:bg-gray-900 shadow-[0_10px_30px_-18px_rgba(15,23,42,0.5)] w-64 transition cursor-pointer hover:-translate-y-0.5 hover:shadow-lg ${
		data?.isIssueTarget
			? 'border-red-400 dark:border-red-600 ring-2 ring-red-200/70 dark:ring-red-800/60 shadow-red-100 dark:shadow-none'
			: data?.hasError
				? 'border-red-200 dark:border-red-800 shadow-red-100 dark:shadow-none'
				: data?.canBeRefined && !data?.isLayer2Child
					? 'border-amber-300 dark:border-amber-700 bg-amber-50/90 dark:bg-amber-950/40 hover:border-amber-400 dark:hover:border-amber-600'
					: data?.canBeRefined || data?.hasLayer2
						? 'border-indigo-200 dark:border-indigo-800 hover:border-indigo-400 dark:hover:border-indigo-600'
						: 'border-gray-200 dark:border-gray-800'
	}`}
	on:click={() => {
		// All clicks are now handled by the parent through onNodeClick
		// The parent will decide whether to expand or show details
		dispatch('select', {
			nodeId: data?.nodeId ?? null,
			title: data?.title ?? '',
			content: data?.detail || data?.content || data?.title || '',
			highlightText: highlightTarget,
			segmentText: data?.segment_text ?? '',
			sectionStart: data?.section_start ?? null,
			sectionEnd: data?.section_end ?? null,
			action: data?.action,
			step: data?.step,
			issues: data?.issues ?? [],
			substeps: data?.substeps ?? [],
			hasError: Boolean(data?.hasError),
			errorDescription: data?.errorDescription ?? null,
			reasoningExcerpt: data?.reasoningExcerpt ?? null,
			// New fields for click behavior
			canBeRefined: data?.canBeRefined ?? false,
			hasLayer2: data?.hasLayer2 ?? false,
			layer2Steps: data?.layer2Steps ?? [],
			layer2Issues: data?.layer2Issues ?? [],
			isLayer2Child: data?.isLayer2Child ?? false,
			parentNodeId: data?.parentNodeId ?? null,
			content_snippet: data?.content_snippet ?? ''
		});
	}}
	on:mouseenter={() => {
		if (isHovering) return; // Already hovering, skip redundant work
		isHovering = true;

		const nodeId = data?.nodeId ?? null;

		// Trigger hover handler immediately for edge highlighting (most critical)
		if (typeof hoverHandler === 'function') {
			hoverHandler(nodeId);
		}

		// Defer less critical events to avoid blocking
		requestAnimationFrame(() => {
			if (!isHovering) return; // User already left, skip

			if (typeof window !== 'undefined') {
				window.dispatchEvent(
					new CustomEvent('reasoning-tree-hover', {
						detail: { nodeId },
						bubbles: false
					})
				);
			}
			dispatch('nodehover', { nodeId });
		});

		// Clear any existing timer
		if (hoverTimer) {
			clearTimeout(hoverTimer);
		}

		// Set 1 second delay before triggering highlight and scroll
		hoverTimer = setTimeout(() => {
			const detail = {
				sentence: highlightTarget,
				sectionStart: data?.section_start ?? null,
				sectionEnd: data?.section_end ?? null
			};

			dispatch('highlight', detail);

			if (typeof window !== 'undefined') {
				window.dispatchEvent(
					new CustomEvent('reasoning-tree-highlight', {
						detail,
						bubbles: false
					})
				);
			}
		}, HOVER_DELAY);
	}}
	on:mouseleave={() => {
		isHovering = false; // Reset hover state

		// Clear the hover timer if mouse leaves before 1 second
		if (hoverTimer) {
			clearTimeout(hoverTimer);
			hoverTimer = null;
		}

		if (typeof hoverHandler === 'function') {
			hoverHandler(null);
		}
		if (typeof window !== 'undefined') {
			window.dispatchEvent(
				new CustomEvent('reasoning-tree-hover', {
					detail: { nodeId: null },
					bubbles: false
				})
			);
		}
		dispatch('nodehover', { nodeId: null });
		dispatch('highlight', { sentence: '' });
	}}
>
	<div class="flex items-center justify-between gap-2 text-[11px] uppercase tracking-wide">
		<div class="flex items-center gap-2 text-gray-500 dark:text-gray-400 font-semibold">
			<span
				class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
				style={`background: ${data?.actionColor ?? '#e2e8f0'}22; color: ${data?.actionColor ?? '#334155'}; border: 1px solid ${data?.actionColor ?? '#cbd5e1'}66;`}
			>
				{actionDisplay(data?.action)}
			</span>
		</div>
		<div class="flex items-center gap-1">
			{#if data?.isLoading}
				<span
					class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-blue-50 text-blue-700 dark:bg-blue-900/50 dark:text-blue-200 border border-blue-200 dark:border-blue-700"
				>
					{$i18n.t('Loading...')}
				</span>
			{:else if data?.isExpanded}
				<span
					class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-green-50 text-green-700 dark:bg-green-900/50 dark:text-green-200 border border-green-200 dark:border-green-700"
				>
					✓
				</span>
			{:else if data?.canBeRefined && !data?.isLayer2Child}
				<!-- Visual indicator that clicking will expand - no button, just subtle indicator -->
				<span
					class="px-1.5 py-0.5 rounded-full text-[10px] text-indigo-500 dark:text-indigo-400"
					title={$i18n.t('Click to see details')}
				>
					⊕
				</span>
			{/if}
			{#if data?.hasError || (data?.issues && data.issues.length > 0)}
				<span
					class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-red-50 text-red-700 dark:bg-red-900/50 dark:text-red-200 border border-red-200 dark:border-red-700"
					title={$i18n.t('Issues detected')}
				>
					!
				</span>
			{/if}
		</div>
	</div>

	<div class="mt-1 text-sm font-semibold text-gray-900 dark:text-gray-100 leading-snug">
		{data?.title ?? $i18n.t('Detail')}
	</div>

	<Handle
		type="target"
		position={targetPosition}
		class="w-2 h-2 rounded-full bg-gray-300 dark:bg-gray-700 border border-white dark:border-gray-900"
	/>
	<Handle
		type="source"
		position={sourcePosition}
		class="w-2 h-2 rounded-full bg-gray-300 dark:bg-gray-700 border border-white dark:border-gray-900"
	/>
</div>
