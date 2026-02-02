<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher, getContext } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { SvelteFlowProvider } from '@xyflow/svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import ReasoningTree from '$lib/components/chat/Messages/ResponseMessage/ReasoningTree.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { analysisStore, type AnalysisTask, models } from '$lib/stores';

	const i18n: any = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let taskId: string;
	export let show = true;

	let task: AnalysisTask | undefined;
	let windowElement: HTMLDivElement;
	let position = { x: 100, y: 100 };
	let size = { width: 800, height: 600 };
	let isDragging = false;
	let isResizing = false;
	let dragStart = { x: 0, y: 0 };
	let dragOrigin = { x: 0, y: 0 };
	let resizeStart = { width: 0, height: 0 };
	let isMinimized = false;
	let isMaximized = false;
	let savedPosition = { ...position };
	let savedSize = { ...size };

	// Status text based on task state
	let statusText = '';

	// Subscribe to task updates
	const unsubscribe = analysisStore.subscribe((state) => {
		task = state.tasks.get(taskId);
		if (task) {
			statusText = getStatusTextInternal(task.status, task.progress);
		}
	});

	function getStatusTextInternal(status: string, progress: any): string {
		switch (status) {
			case 'pending':
				return 'Waiting to start...';
			case 'running':
				return 'Starting analysis...';
			case 'layer1':
				return 'Analyzing reasoning structure...';
			case 'layer2':
				return `Analyzing details... (${progress?.layer2Completed || 0}/${progress?.layer2Total || '?'})`;
			case 'error_detection':
				return 'Detecting errors...';
			case 'complete':
				return 'Analysis complete';
			case 'error':
				return 'Analysis failed';
			case 'cancelled':
				return 'Analysis cancelled';
			default:
				return '';
		}
	}

	onDestroy(() => {
		unsubscribe();
	});

	onMount(() => {
		// Center the window on screen
		if (typeof window !== 'undefined') {
			position = {
				x: Math.max(50, (window.innerWidth - size.width) / 2),
				y: Math.max(50, (window.innerHeight - size.height) / 2)
			};
		}
	});

	// Window dragging
	function startDrag(e: PointerEvent) {
		if (isMaximized) return;
		isDragging = true;
		dragStart = { x: e.clientX, y: e.clientY };
		dragOrigin = { ...position };
		window.addEventListener('pointermove', handleDrag);
		window.addEventListener('pointerup', stopDrag);
	}

	function handleDrag(e: PointerEvent) {
		if (!isDragging) return;
		const dx = e.clientX - dragStart.x;
		const dy = e.clientY - dragStart.y;
		position = {
			x: Math.max(0, Math.min(dragOrigin.x + dx, window.innerWidth - 100)),
			y: Math.max(0, Math.min(dragOrigin.y + dy, window.innerHeight - 50))
		};
	}

	function stopDrag() {
		isDragging = false;
		window.removeEventListener('pointermove', handleDrag);
		window.removeEventListener('pointerup', stopDrag);
	}

	// Window resizing
	function startResize(e: PointerEvent) {
		if (isMaximized) return;
		isResizing = true;
		dragStart = { x: e.clientX, y: e.clientY };
		resizeStart = { ...size };
		window.addEventListener('pointermove', handleResize);
		window.addEventListener('pointerup', stopResize);
		e.stopPropagation();
	}

	function handleResize(e: PointerEvent) {
		if (!isResizing) return;
		const dx = e.clientX - dragStart.x;
		const dy = e.clientY - dragStart.y;
		size = {
			width: Math.max(400, resizeStart.width + dx),
			height: Math.max(300, resizeStart.height + dy)
		};
	}

	function stopResize() {
		isResizing = false;
		window.removeEventListener('pointermove', handleResize);
		window.removeEventListener('pointerup', stopResize);
	}

	// Window actions
	function minimize() {
		isMinimized = !isMinimized;
	}

	function maximize() {
		if (isMaximized) {
			position = { ...savedPosition };
			size = { ...savedSize };
			isMaximized = false;
		} else {
			savedPosition = { ...position };
			savedSize = { ...size };
			position = { x: 0, y: 0 };
			size = { width: window.innerWidth, height: window.innerHeight };
			isMaximized = true;
		}
	}

	function close() {
		show = false;
		dispatch('close');
	}

	function cancelAnalysis() {
		if (task) {
			analysisStore.cancelTask(taskId);
		}
	}

	// Check if task is still in progress
	$: isInProgress =
		task &&
		(task.status === 'pending' ||
			task.status === 'running' ||
			task.status === 'layer1' ||
			task.status === 'layer2' ||
			task.status === 'error_detection');

	// Progress percentage
	$: progressPercent = task?.progress?.percentComplete ?? 0;
</script>

{#if show && task}
	<!-- Backdrop -->
	<div
		class="fixed inset-0 bg-black/20 dark:bg-black/40 z-[998]"
		on:click={close}
		on:keydown={(e) => e.key === 'Escape' && close()}
		role="button"
		tabindex="-1"
		aria-label="Close analysis window"
		transition:fade={{ duration: 150 }}
	></div>

	<!-- Window -->
	<div
		bind:this={windowElement}
		class="fixed z-[999] bg-white dark:bg-gray-900 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden"
		class:transition-all={!isDragging && !isResizing}
		class:duration-200={!isDragging && !isResizing}
		style="
			left: {position.x}px;
			top: {position.y}px;
			width: {size.width}px;
			height: {isMinimized ? '44px' : size.height + 'px'};
		"
		transition:fly={{ y: 20, duration: 200 }}
	>
		<!-- Title Bar -->
		<div
			class="flex items-center justify-between px-4 py-2 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-850 border-b border-gray-200 dark:border-gray-700 cursor-move select-none"
			on:pointerdown={startDrag}
			role="toolbar"
		>
			<div class="flex items-center gap-3 min-w-0 flex-1">
				<!-- Status indicator -->
				{#if isInProgress}
					<div class="flex-shrink-0">
						<Spinner className="size-4" />
					</div>
				{:else if task.status === 'complete'}
					<div class="size-3 rounded-full bg-green-500 flex-shrink-0"></div>
				{:else if task.status === 'error'}
					<div class="size-3 rounded-full bg-red-500 flex-shrink-0"></div>
				{:else if task.status === 'cancelled'}
					<div class="size-3 rounded-full bg-yellow-500 flex-shrink-0"></div>
				{/if}

				<div class="min-w-0 flex-1">
					<div class="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
						{$i18n.t('Error Analysis')} - {task.chatTitle || $i18n.t('Chat')}
					</div>
					<div class="text-xs text-gray-500 dark:text-gray-400 truncate">
						{statusText}
						{#if task.modelName}
							<span class="mx-1">•</span>
							<span>{task.modelName}</span>
						{/if}
					</div>
				</div>
			</div>

			<div class="flex items-center gap-1 ml-2">
				<!-- Cancel button (only when in progress) -->
				{#if isInProgress}
					<Tooltip content={$i18n.t('Cancel')}>
						<button
							class="p-1.5 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
							on:click|stopPropagation={cancelAnalysis}
							aria-label="Cancel analysis"
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								class="size-4"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
							>
								<rect x="3" y="3" width="18" height="18" rx="2" />
							</svg>
						</button>
					</Tooltip>
				{/if}

				<!-- Minimize -->
				<button
					class="p-1.5 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
					on:click|stopPropagation={minimize}
					aria-label="Minimize window"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="size-4"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
					>
						<line x1="5" y1="12" x2="19" y2="12" />
					</svg>
				</button>

				<!-- Maximize -->
				<button
					class="p-1.5 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
					on:click|stopPropagation={maximize}
					aria-label={isMaximized ? 'Restore window' : 'Maximize window'}
				>
					{#if isMaximized}
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="size-4"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
						>
							<path
								d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"
							/>
						</svg>
					{:else}
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="size-4"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
						>
							<rect x="3" y="3" width="18" height="18" rx="2" />
						</svg>
					{/if}
				</button>

				<!-- Close -->
				<button
					class="p-1.5 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/30 text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400"
					on:click|stopPropagation={close}
					aria-label="Close window"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="size-4"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
					>
						<line x1="18" y1="6" x2="6" y2="18" />
						<line x1="6" y1="6" x2="18" y2="18" />
					</svg>
				</button>
			</div>
		</div>

		<!-- Progress bar -->
		{#if isInProgress && !isMinimized}
			<div class="h-1 w-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
				<div
					class="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-300 ease-out"
					style="width: {progressPercent}%"
				></div>
			</div>
		{/if}

		<!-- Content -->
		{#if !isMinimized}
			<div class="flex-1 overflow-y-auto p-4">
				{#if isInProgress && !task.result}
					<!-- Loading state -->
					<div class="w-full max-w-xl mx-auto" aria-live="polite">
						<div
							class="rounded-2xl border border-gray-200/80 dark:border-gray-800/80 bg-gradient-to-b from-gray-50 to-white dark:from-gray-900/40 dark:to-gray-900/60 p-5 shadow-sm space-y-4"
						>
							<div class="flex items-center gap-3">
								<div
									class="p-2 rounded-full bg-white/90 dark:bg-gray-800/80 shadow-sm border border-gray-200/70 dark:border-gray-800/70"
								>
									<Spinner className="size-5 text-gray-700 dark:text-gray-200" />
								</div>
								<div class="flex-1 min-w-0">
									<div class="text-sm font-semibold truncate">{statusText}</div>
									<div class="text-xs text-gray-500 dark:text-gray-400">
										{$i18n.t(
											'You can continue using other conversations while this analysis runs.'
										)}
									</div>
								</div>
								<div
									class="text-[11px] font-semibold text-gray-500 dark:text-gray-400 pl-3 border-l border-gray-200 dark:border-gray-800"
								>
									{Math.round(progressPercent)}%
								</div>
							</div>

							<div class="h-2 w-full rounded-full bg-gray-200/80 dark:bg-gray-800 overflow-hidden">
								<div
									class="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-300 ease-out"
									style="width: {progressPercent}%"
								></div>
							</div>

							<div class="grid grid-cols-3 gap-3 text-xs text-gray-600 dark:text-gray-300">
								<div class="flex items-center gap-2">
									<div
										class="size-2.5 rounded-full {task.status === 'layer1'
											? 'bg-blue-500 animate-pulse'
											: task.status === 'layer2' ||
												  task.status === 'error_detection' ||
												  task.status === 'complete'
												? 'bg-green-500'
												: 'bg-gray-300 dark:bg-gray-700'}"
									></div>
									<div class="truncate">{$i18n.t('Structure')}</div>
								</div>
								<div class="flex items-center gap-2">
									<div
										class="size-2.5 rounded-full {task.status === 'layer2'
											? 'bg-indigo-500 animate-pulse'
											: task.status === 'error_detection' || task.status === 'complete'
												? 'bg-green-500'
												: 'bg-gray-300 dark:bg-gray-700'}"
									></div>
									<div class="truncate">{$i18n.t('Details')}</div>
								</div>
								<div class="flex items-center gap-2">
									<div
										class="size-2.5 rounded-full {task.status === 'error_detection'
											? 'bg-purple-500 animate-pulse'
											: task.status === 'complete'
												? 'bg-green-500'
												: 'bg-gray-300 dark:bg-gray-700'}"
									></div>
									<div class="truncate">{$i18n.t('Errors')}</div>
								</div>
							</div>
						</div>
					</div>
				{:else if task.status === 'error'}
					<!-- Error state -->
					<div
						class="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/40 text-red-700 dark:text-red-200 p-4"
					>
						<div class="font-medium mb-1">{$i18n.t('Analysis Failed')}</div>
						<div class="text-sm">{task.error || $i18n.t('An unknown error occurred')}</div>
					</div>
				{:else if task.status === 'cancelled'}
					<!-- Cancelled state -->
					<div
						class="rounded-lg border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-200 p-4"
					>
						<div class="font-medium">{$i18n.t('Analysis Cancelled')}</div>
					</div>
				{:else if task.result}
					<!-- Results -->
					<div class="space-y-3 text-sm">
						<SvelteFlowProvider>
							<ReasoningTree
								nodes={task.result.nodes ?? []}
								edges={task.result.edges ?? []}
								sections={task.result.sections ?? []}
								overthinkingAnalysis={task.errorDetection?.overthinking_analysis ?? null}
								detectedErrors={task.errorDetection?.errors ?? []}
								chatId={task.chatId}
								messageId={task.messageId}
								model={task?.model ?? undefined}
								analysisStage={task.status}
							/>
						</SvelteFlowProvider>
					</div>
				{:else}
					<div class="text-sm text-gray-600 dark:text-gray-300 text-center py-8">
						{$i18n.t('No analysis data available')}
					</div>
				{/if}
			</div>
		{/if}

		<!-- Resize handle -->
		{#if !isMaximized && !isMinimized}
			<div
				class="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize"
				on:pointerdown={startResize}
				role="separator"
				aria-orientation="horizontal"
			>
				<svg class="w-full h-full text-gray-400" viewBox="0 0 16 16">
					<path d="M14 14L6 14L14 6L14 14Z" fill="currentColor" opacity="0.3" />
					<path d="M14 14L10 14L14 10L14 14Z" fill="currentColor" opacity="0.5" />
				</svg>
			</div>
		{/if}
	</div>
{/if}

<style>
	/* Prevent text selection during drag */
	.cursor-move {
		user-select: none;
	}
</style>
