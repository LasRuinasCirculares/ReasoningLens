<script lang="ts">
	/**
	 * ModelAnalysisReport Component
	 *
	 * Displays aggregated analysis statistics and generates comprehensive
	 * reports for reasoning models based on multiple analysis sessions.
	 */
	import { createEventDispatcher, getContext, onMount } from 'svelte';
	import { fade, slide } from 'svelte/transition';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Markdown from '$lib/components/chat/Messages/Markdown.svelte';
	import {
		getAnalysisHistory,
		getAggregatedStats,
		generateAnalysisReport,
		type AnalysisHistoryRecord,
		type AggregatedStats,
		type ReportGenerationEvent
	} from '$lib/apis/analysis';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	// Props
	export let reasoningModel: string = '';
	export let analysisModel: string = '';
	export let availableModels: Array<{ id: string; name: string }> = [];

	// State
	let loadingHistory = false;
	let generatingReport = false;
	let error: string | null = null;

	// Data
	let historyRecords: AnalysisHistoryRecord[] = [];
	let stats: AggregatedStats | null = null;
	let report: string = '';
	let displayContent: string = ''; // Content for rendering

	// UI State
	let showHistory = false;
	let selectedReportModel: string = '';
	let progressMessage: string = '';

	// Streaming - use array to avoid expensive string concatenation
	let contentChunks: string[] = [];
	let lastDisplayLength = 0;
	let updateTimer: ReturnType<typeof setTimeout> | null = null;
	const UPDATE_INTERVAL_MS = 1500; // Update display every 1.5s
	let progressCleared = false;

	// Function to get current content as string
	function getCurrentContent(): string {
		return contentChunks.join('');
	}

	// Function to update display with minimal work
	function updateDisplay() {
		const currentLen = contentChunks.reduce((acc, c) => acc + c.length, 0);
		if (currentLen > lastDisplayLength + 100) {
			displayContent = getCurrentContent();
			lastDisplayLength = currentLen;
		}
	}

	// Function to schedule an update (heavily debounced)
	function scheduleUpdate() {
		if (!updateTimer) {
			updateTimer = setTimeout(() => {
				updateDisplay();
				updateTimer = null;
			}, UPDATE_INTERVAL_MS);
		}
	}

	// Function to force immediate update (for completion)
	function forceImmediateUpdate() {
		if (updateTimer) {
			clearTimeout(updateTimer);
			updateTimer = null;
		}
		report = getCurrentContent();
		displayContent = report;
		lastDisplayLength = report.length;
	}

	// Computed
	$: selectedRecords = reasoningModel
		? historyRecords.filter((r) => r.reasoning_model.includes(reasoningModel))
		: historyRecords;

	onMount(async () => {
		await loadHistory();
	});

	async function loadHistory() {
		loadingHistory = true;
		error = null;

		try {
			const response = await getAnalysisHistory(localStorage.token, {
				reasoning_model: reasoningModel || undefined,
				analysis_model: analysisModel || undefined,
				limit: 100
			});

			historyRecords = response.records;

			// Also load stats
			if (historyRecords.length > 0) {
				const statsResponse = await getAggregatedStats(localStorage.token, {
					reasoning_model: reasoningModel || undefined,
					analysis_model: analysisModel || undefined
				});
				stats = statsResponse.stats;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			console.error('Failed to load analysis history:', e);
		} finally {
			loadingHistory = false;
		}
	}

	async function handleGenerateReport() {
		if (!selectedReportModel) {
			error = $i18n.t('Please select a model for generating the report');
			return;
		}

		generatingReport = true;
		error = null;
		report = '';
		displayContent = '';
		contentChunks = []; // Reset content chunks
		lastDisplayLength = 0;
		progressMessage = $i18n.t('Initializing...');

		try {
			const result = await generateAnalysisReport(
				localStorage.token,
				{
					reasoning_model: reasoningModel || 'all',
					analysis_model: analysisModel || undefined,
					report_model: selectedReportModel,
					stream: true
				},
				(event: ReportGenerationEvent) => {
					if (event.type === 'progress') {
						progressMessage = event.message || '';
						progressCleared = false;
					} else if (event.type === 'chunk' && event.content) {
						// Just push to array - very fast
						contentChunks.push(event.content);
						scheduleUpdate();
						if (!progressCleared) {
							progressMessage = '';
							progressCleared = true;
						}
					} else if (event.type === 'complete') {
						if (event.report) {
							contentChunks = [event.report];
						}
						forceImmediateUpdate();
						if (event.stats) stats = event.stats;
					}
				}
			);

			if (result.report) {
				contentChunks = [result.report];
			}
			forceImmediateUpdate();
			stats = result.stats;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			console.error('Failed to generate report:', e);
		} finally {
			generatingReport = false;
			progressMessage = '';
			progressCleared = false;
			if (contentChunks.length > 0) {
				forceImmediateUpdate();
			}
		}
	}

	function formatPercentage(value: number): string {
		return (value * 100).toFixed(1) + '%';
	}

	function getErrorTypeColor(type: string): string {
		const colors: Record<string, string> = {
			'Knowledge Error': '#4338ca',
			'Logical Error': '#e11d48',
			'Formal Error': '#0ea5e9',
			Safety: '#c2410c',
			Hallucination: '#0f766e',
			Readability: '#374151',
			Overthinking: '#6b21a8'
		};
		return colors[type] || '#6b7280';
	}
</script>

<div class="model-analysis-report">
	<!-- Header -->
	<div class="flex items-center justify-between mb-4">
		<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
			{$i18n.t('Model Analysis Report')}
		</h3>
		<button
			class="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
			on:click={() => dispatch('close')}
		>
			<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M6 18L18 6M6 6l12 12"
				/>
			</svg>
		</button>
	</div>

	<!-- Model Selection / Filter -->
	<div class="grid grid-cols-2 gap-4 mb-4">
		<div>
			<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
				{$i18n.t('Reasoning Model (being analyzed)')}
			</label>
			<input
				type="text"
				bind:value={reasoningModel}
				placeholder={$i18n.t('Filter by model name...')}
				class="w-full px-3 py-2 text-sm border rounded-lg dark:bg-gray-800 dark:border-gray-700 dark:text-gray-100"
				on:change={loadHistory}
			/>
		</div>
		<div>
			<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
				{$i18n.t('Analysis Model (used for analysis)')}
			</label>
			<input
				type="text"
				bind:value={analysisModel}
				placeholder={$i18n.t('Filter by model name...')}
				class="w-full px-3 py-2 text-sm border rounded-lg dark:bg-gray-800 dark:border-gray-700 dark:text-gray-100"
				on:change={loadHistory}
			/>
		</div>
	</div>

	{#if loadingHistory}
		<div class="flex items-center justify-center py-8">
			<Spinner className="w-6 h-6" />
			<span class="ml-2 text-gray-500">{$i18n.t('Loading analysis history...')}</span>
		</div>
	{:else if error && !stats && !report}
		<div
			class="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/30 p-4 text-red-700 dark:text-red-300"
		>
			{error}
		</div>
	{:else if historyRecords.length === 0}
		<div class="text-center py-8 text-gray-500 dark:text-gray-400">
			<svg
				class="w-12 h-12 mx-auto mb-4 text-gray-300 dark:text-gray-600"
				fill="none"
				stroke="currentColor"
				viewBox="0 0 24 24"
			>
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="1.5"
					d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
				/>
			</svg>
			<p>{$i18n.t('No analysis records found')}</p>
			<p class="text-sm mt-1">
				{$i18n.t('Run some reasoning analyses first to generate a report')}
			</p>
		</div>
	{:else}
		<!-- Statistics Overview -->
		{#if stats}
			<div class="grid grid-cols-4 gap-3 mb-4" transition:fade>
				<div class="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-3">
					<div class="text-2xl font-bold text-blue-600 dark:text-blue-400">
						{stats.total_analyses}
					</div>
					<div class="text-xs text-blue-600/70 dark:text-blue-400/70">{$i18n.t('Analyses')}</div>
				</div>
				<div class="bg-red-50 dark:bg-red-900/30 rounded-lg p-3">
					<div class="text-2xl font-bold text-red-600 dark:text-red-400">{stats.total_errors}</div>
					<div class="text-xs text-red-600/70 dark:text-red-400/70">{$i18n.t('Errors')}</div>
				</div>
				<div class="bg-purple-50 dark:bg-purple-900/30 rounded-lg p-3">
					<div class="text-2xl font-bold text-purple-600 dark:text-purple-400">
						{formatPercentage(stats.avg_overthinking_score)}
					</div>
					<div class="text-xs text-purple-600/70 dark:text-purple-400/70">
						{$i18n.t('Avg Overthinking')}
					</div>
				</div>
				<div class="bg-green-50 dark:bg-green-900/30 rounded-lg p-3">
					<div class="text-2xl font-bold text-green-600 dark:text-green-400">
						{stats.avg_sections.toFixed(1)}
					</div>
					<div class="text-xs text-green-600/70 dark:text-green-400/70">
						{$i18n.t('Avg Sections')}
					</div>
				</div>
			</div>

			<!-- Error Type Distribution -->
			{#if Object.keys(stats.error_type_counts).length > 0}
				<div class="mb-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg" transition:slide>
					<h4 class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
						{$i18n.t('Error Distribution')}
					</h4>
					<div class="space-y-2">
						{#each Object.entries(stats.error_type_counts).sort((a, b) => b[1] - a[1]) as [type, count]}
							{@const percentage = (count / stats.total_errors) * 100}
							<div class="flex items-center gap-2">
								<div class="w-28 text-xs text-gray-600 dark:text-gray-400 truncate" title={type}>
									{type}
								</div>
								<div class="flex-1 h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
									<div
										class="h-full rounded-full transition-all duration-300"
										style="width: {percentage}%; background-color: {getErrorTypeColor(type)}"
									></div>
								</div>
								<div class="w-16 text-xs text-right text-gray-500">
									{count} ({percentage.toFixed(0)}%)
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		{/if}

		<!-- History Toggle -->
		<button
			class="w-full flex items-center justify-between p-3 mb-4 text-sm bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
			on:click={() => (showHistory = !showHistory)}
		>
			<span class="font-medium text-gray-700 dark:text-gray-300">
				{$i18n.t('Analysis History')} ({selectedRecords.length})
			</span>
			<svg
				class="w-5 h-5 text-gray-400 transition-transform"
				class:rotate-180={showHistory}
				fill="none"
				stroke="currentColor"
				viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
			</svg>
		</button>

		{#if showHistory}
			<div
				class="max-h-48 overflow-y-auto mb-4 border rounded-lg dark:border-gray-700"
				transition:slide
			>
				{#each selectedRecords as record}
					<div
						class="flex items-center justify-between p-2 border-b dark:border-gray-700 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-800/50"
					>
						<div class="flex-1 min-w-0">
							<div class="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
								{record.query_preview || 'No query'}
							</div>
							<div class="text-xs text-gray-500 dark:text-gray-400">
								{record.reasoning_model} • {new Date(record.timestamp).toLocaleDateString()}
							</div>
						</div>
						<div class="flex items-center gap-2 ml-2">
							<span
								class="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300"
							>
								{record.section_count} sections
							</span>
							{#if record.error_count > 0}
								<span
									class="px-2 py-0.5 text-xs rounded-full bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400"
								>
									{record.error_count} errors
								</span>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Report Generation -->
		<div class="border-t dark:border-gray-700 pt-4">
			<h4 class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
				{$i18n.t('Generate Comprehensive Report')}
			</h4>

			<div class="flex gap-3 mb-4">
				<select
					bind:value={selectedReportModel}
					class="flex-1 px-3 py-2 text-sm border rounded-lg dark:bg-gray-800 dark:border-gray-700 dark:text-gray-100"
				>
					<option value="">{$i18n.t('Select report generation model...')}</option>
					{#each availableModels as model}
						<option value={model.id}>{model.name}</option>
					{/each}
				</select>

				<button
					class="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
					on:click={handleGenerateReport}
					disabled={generatingReport || !selectedReportModel || selectedRecords.length === 0}
				>
					{#if generatingReport}
						<Spinner className="w-4 h-4" />
						{$i18n.t('Generating...')}
					{:else}
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
							/>
						</svg>
						{$i18n.t('Generate Report')}
					{/if}
				</button>
			</div>

			{#if progressMessage}
				<div class="text-sm text-gray-500 dark:text-gray-400 mb-3" transition:fade>
					{progressMessage}
				</div>
			{/if}

			{#if error && (report || displayContent)}
				<div
					class="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/30 p-3 mb-4 text-sm text-red-700 dark:text-red-300"
				>
					{error}
				</div>
			{/if}

			<!-- Report Content -->
			{#if generatingReport && displayContent}
				<!-- During streaming: show plain text for performance -->
				<div
					class="prose prose-sm dark:prose-invert max-w-none p-4 bg-white dark:bg-gray-900 border rounded-lg dark:border-gray-700 max-h-96 overflow-y-auto"
					transition:slide
				>
					<pre
						class="whitespace-pre-wrap font-sans text-sm text-gray-800 dark:text-gray-200 bg-transparent p-0 m-0 overflow-visible">{displayContent}</pre>
					<!-- Typing cursor when streaming -->
					<span
						class="inline-block w-1.5 h-4 ml-0.5 -mb-0.5 bg-gray-400 dark:bg-gray-500 animate-pulse rounded-sm"
					></span>
				</div>
			{:else if report || displayContent}
				<!-- After completion: render full Markdown -->
				<div
					class="prose prose-sm dark:prose-invert max-w-none p-4 bg-white dark:bg-gray-900 border rounded-lg dark:border-gray-700 max-h-96 overflow-y-auto"
					transition:slide
				>
					<Markdown id="analysis-report" content={report || displayContent} done={true} />
				</div>
			{/if}

			{#if report || displayContent}
				<!-- Export Options -->
				<div class="flex justify-end gap-2 mt-3">
					<button
						class="px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
						on:click={() => {
							const blob = new Blob([report || displayContent], { type: 'text/markdown' });
							const url = URL.createObjectURL(blob);
							const a = document.createElement('a');
							a.href = url;
							a.download = `analysis-report-${Date.now()}.md`;
							a.click();
							URL.revokeObjectURL(url);
						}}
					>
						<svg class="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
							/>
						</svg>
						{$i18n.t('Download Markdown')}
					</button>
					<button
						class="px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
						on:click={() => {
							navigator.clipboard.writeText(report || displayContent);
						}}
					>
						<svg class="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
							/>
						</svg>
						{$i18n.t('Copy')}
					</button>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.model-analysis-report {
		padding: 1rem;
	}

	.prose :global(h1) {
		font-size: 1.25rem;
		font-weight: 700;
		margin-bottom: 1rem;
	}

	.prose :global(h2) {
		font-size: 1.125rem;
		font-weight: 600;
		margin-bottom: 0.75rem;
		margin-top: 1rem;
	}

	.prose :global(h3) {
		font-size: 1rem;
		font-weight: 500;
		margin-bottom: 0.5rem;
		margin-top: 0.75rem;
	}

	.prose :global(ul) {
		list-style-type: disc;
		padding-left: 1.25rem;
		margin-bottom: 0.75rem;
	}

	.prose :global(ol) {
		list-style-type: decimal;
		padding-left: 1.25rem;
		margin-bottom: 0.75rem;
	}

	.prose :global(li) {
		margin-bottom: 0.25rem;
	}

	.prose :global(p) {
		margin-bottom: 0.5rem;
	}

	.prose :global(strong) {
		font-weight: 600;
	}
</style>
