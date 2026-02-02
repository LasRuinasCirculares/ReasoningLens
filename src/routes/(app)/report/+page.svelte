<script lang="ts">
	/**
	 * Model Analysis Report Page
	 *
	 * Chat-like interface for generating model analysis reports.
	 * Similar to the main chat window design with streaming output.
	 */
	import { getContext, onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import { goto } from '$app/navigation';
	import { models, mobile, showSidebar, settings, config } from '$lib/stores';
	import { getModels } from '$lib/apis';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Markdown from '$lib/components/chat/Messages/Markdown.svelte';
	import Skeleton from '$lib/components/chat/Messages/Skeleton.svelte';
	import {
		getAnalysisHistory,
		getAggregatedStats,
		generateAnalysisReport,
		listSavedReports,
		getSavedReport,
		deleteSavedReport,
		type AnalysisHistoryRecord,
		type AggregatedStats,
		type ReportGenerationEvent,
		type SavedReportSummary,
		type SavedReportFull
	} from '$lib/apis/analysis';

	const i18n = getContext('i18n');

	// Model selection
	let selectedReasoningModel: string = '';
	let selectedReportModel: string = '';

	// Language selection for report generation
	let selectedLanguage: 'zh' | 'en' = 'zh';

	// State
	let loading = true;
	let generatingReport = false;
	let error: string | null = null;
	let loadingSavedReports = false;

	// Data
	let historyRecords: AnalysisHistoryRecord[] = [];
	let stats: AggregatedStats | null = null;
	let savedReports: SavedReportSummary[] = [];
	let loadedReportId: string | null = null; // Track which saved report is currently displayed

	// Report message (like a chat message)
	let reportContent: string = ''; // Final content (only set on complete)
	let displayContent: string = ''; // Content for rendering
	let progressMessage: string = '';

	// Streaming buffer - use array to avoid expensive string concatenation
	let contentChunks: string[] = []; // All chunks accumulated
	let lastDisplayLength = 0; // Track what's been displayed
	let updateTimer: ReturnType<typeof setTimeout> | null = null;
	const UPDATE_INTERVAL_MS = 1500; // Update display every 1.5s (very low frequency)
	let progressCleared = false;
	let isScrolling = false; // Prevent scroll during animation

	// Function to get current content as string (only when needed)
	function getCurrentContent(): string {
		return contentChunks.join('');
	}

	// Function to update display with minimal work
	function updateDisplay() {
		const currentLen = contentChunks.reduce((acc, c) => acc + c.length, 0);
		// Only update if we have significant new content
		if (currentLen > lastDisplayLength + 100) {
			displayContent = getCurrentContent();
			lastDisplayLength = currentLen;
			// Manual scroll after a short delay
			if (!isScrolling && messagesContainer) {
				isScrolling = true;
				setTimeout(() => {
					if (messagesContainer) {
						messagesContainer.scrollTop = messagesContainer.scrollHeight;
					}
					isScrolling = false;
				}, 50);
			}
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
		reportContent = getCurrentContent();
		displayContent = reportContent;
		lastDisplayLength = reportContent.length;
		// Final scroll
		if (messagesContainer) {
			setTimeout(() => {
				if (messagesContainer) {
					messagesContainer.scrollTop = messagesContainer.scrollHeight;
				}
			}, 100);
		}
	}

	// Scroll container ref
	let messagesContainer: HTMLDivElement;

	// Convert models store to the format for selection
	$: availableModels = ($models ?? []).map((m) => ({
		id: m.id,
		name: m.name || m.id
	}));

	// Count analyses per reasoning model
	$: modelAnalysisCounts = historyRecords.reduce(
		(acc, r) => {
			if (r.reasoning_model && r.reasoning_model !== 'unknown') {
				acc[r.reasoning_model] = (acc[r.reasoning_model] || 0) + 1;
			}
			return acc;
		},
		{} as Record<string, number>
	);

	// Get reasoning models with at least MIN_ANALYSES_FOR_REPORT analysis records (eligible for report generation)
	const MIN_ANALYSES_FOR_REPORT = 3;
	$: eligibleReasoningModels = Object.entries(modelAnalysisCounts)
		.filter(([_, count]) => count >= MIN_ANALYSES_FOR_REPORT)
		.map(([model, count]) => ({ model, count }))
		.sort((a, b) => b.count - a.count);

	// Load history on mount
	async function loadHistory() {
		loading = true;
		error = null;

		try {
			const response = await getAnalysisHistory(localStorage.token, {
				limit: 200
			});
			historyRecords = response.records;

			// Load stats
			if (historyRecords.length > 0) {
				const statsResponse = await getAggregatedStats(localStorage.token, {});
				stats = statsResponse.stats;
			}

			// Also load saved reports
			await loadSavedReports();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			console.error('Failed to load analysis history:', e);
		} finally {
			loading = false;
		}
	}

	// Load saved reports list
	async function loadSavedReports() {
		loadingSavedReports = true;
		try {
			const response = await listSavedReports(localStorage.token, { limit: 50 });
			savedReports = response.reports;
		} catch (e) {
			console.error('Failed to load saved reports:', e);
		} finally {
			loadingSavedReports = false;
		}
	}

	// Load a specific saved report
	async function handleLoadSavedReport(reportId: string) {
		try {
			loading = true;
			error = null;

			const response = await getSavedReport(localStorage.token, reportId);
			const report = response.report;

			// Display the loaded report
			reportContent = report.report_content;
			displayContent = report.report_content;
			contentChunks = [report.report_content];
			lastDisplayLength = report.report_content.length;
			loadedReportId = reportId;

			// Update model selections to match the loaded report
			selectedReasoningModel = report.reasoning_model;
			selectedLanguage = report.language as 'zh' | 'en';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			console.error('Failed to load saved report:', e);
		} finally {
			loading = false;
		}
	}

	// Delete a saved report
	async function handleDeleteSavedReport(reportId: string) {
		if (!confirm($i18n.t('Are you sure you want to delete this report?'))) {
			return;
		}

		try {
			await deleteSavedReport(localStorage.token, reportId);
			// Refresh the list
			await loadSavedReports();

			// If we deleted the currently displayed report, clear it
			if (loadedReportId === reportId) {
				reportContent = '';
				displayContent = '';
				loadedReportId = null;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			console.error('Failed to delete saved report:', e);
		}
	}

	// Format timestamp for display
	function formatTimestamp(timestamp: string): string {
		try {
			const date = new Date(timestamp);
			return date.toLocaleString();
		} catch {
			return timestamp;
		}
	}

	// Load on mount
	onMount(async () => {
		// Ensure models are loaded
		if ($models.length === 0) {
			try {
				const loadedModels = await getModels(
					localStorage.token,
					$config?.features?.enable_direct_connections
						? ($settings?.directConnections ?? null)
						: null
				);
				models.set(loadedModels);
			} catch (e) {
				console.error('Failed to load models:', e);
			}
		}
		await loadHistory();
	});

	async function handleGenerateReport() {
		if (!selectedReasoningModel) {
			error = $i18n.t('Please select a reasoning model to analyze');
			return;
		}
		if (!selectedReportModel) {
			error = $i18n.t('Please select a model for generating the report');
			return;
		}

		generatingReport = true;
		error = null;
		reportContent = '';
		displayContent = '';
		contentChunks = []; // Reset content chunks
		lastDisplayLength = 0;
		progressMessage = $i18n.t('Collecting analysis data...');
		loadedReportId = null; // Clear loaded report tracking

		try {
			await generateAnalysisReport(
				localStorage.token,
				{
					reasoning_model: selectedReasoningModel,
					report_model: selectedReportModel,
					language: selectedLanguage,
					stream: true
				},
				(event: ReportGenerationEvent) => {
					if (event.type === 'progress') {
						progressMessage = event.message || '';
						progressCleared = false;
					} else if (event.type === 'chunk' && event.content) {
						// Just push to array - very fast, no string concatenation
						contentChunks.push(event.content);
						scheduleUpdate();
						// Only clear progress once
						if (!progressCleared) {
							progressMessage = '';
							progressCleared = true;
						}
					} else if (event.type === 'complete') {
						if (event.report) {
							contentChunks = [event.report]; // Replace with final
						}
						forceImmediateUpdate();
						if (event.stats) stats = event.stats;
					} else if (event.type === 'error') {
						error = event.message || $i18n.t('Failed to generate report');
					}
				}
			);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			console.error('Failed to generate report:', e);
		} finally {
			generatingReport = false;
			progressMessage = '';
			progressCleared = false;
			// Ensure final content is displayed
			if (contentChunks.length > 0) {
				forceImmediateUpdate();
			}
			// Refresh saved reports list after generation
			await loadSavedReports();
		}
	}

	function handleExport() {
		if (!reportContent) return;
		const blob = new Blob([reportContent], { type: 'text/markdown' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `model-analysis-report-${Date.now()}.md`;
		a.click();
		URL.revokeObjectURL(url);
	}

	function handleCopy() {
		if (!reportContent) return;
		navigator.clipboard.writeText(reportContent);
	}
</script>

<div class="h-screen max-h-[100dvh] w-full flex flex-col bg-white dark:bg-gray-900">
	<!-- Top bar with model selectors (similar to chat top bar) -->
	<div
		class="flex-shrink-0 border-b border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900"
	>
		<div class="max-w-5xl mx-auto px-4 py-3">
			<div class="flex flex-wrap items-center gap-4">
				<!-- Title -->
				<div class="flex items-center gap-3">
					<div class="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600">
						<svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
							/>
						</svg>
					</div>
					<h1 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
						{$i18n.t('Model Analysis Report')}
					</h1>
				</div>

				<div class="flex-1"></div>

				<!-- Model selectors -->
				<div class="flex flex-wrap items-center gap-3">
					<!-- Reasoning Model selector -->
					<div class="flex items-center gap-2">
						<span
							class="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap hidden sm:inline"
						>
							{$i18n.t('Reasoning Model')}:
						</span>
						<select
							bind:value={selectedReasoningModel}
							class="px-3 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 dark:text-gray-100 min-w-[180px]"
							aria-label={$i18n.t('Reasoning Model')}
						>
							<option value="">{$i18n.t('Select reasoning model...')}</option>
							{#each eligibleReasoningModels as { model, count }}
								<option value={model}>{model} ({count} {$i18n.t('analyses')})</option>
							{/each}
						</select>
					</div>

					<!-- Report Generation Model selector -->
					<div class="flex items-center gap-2">
						<span
							class="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap hidden sm:inline"
						>
							{$i18n.t('Report Model')}:
						</span>
						<select
							bind:value={selectedReportModel}
							class="px-3 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 dark:text-gray-100 min-w-[140px]"
							aria-label={$i18n.t('Report Model')}
						>
							<option value="">{$i18n.t('Select model...')}</option>
							{#each availableModels as model}
								<option value={model.id}>{model.name}</option>
							{/each}
						</select>
					</div>

					<!-- Language selector -->
					<div class="flex items-center gap-2">
						<span
							class="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap hidden sm:inline"
						>
							{$i18n.t('Report Language')}:
						</span>
						<select
							bind:value={selectedLanguage}
							class="px-3 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 dark:text-gray-100 min-w-[100px]"
							aria-label={$i18n.t('Report Language')}
						>
							<option value="zh">中文</option>
							<option value="en">English</option>
						</select>
					</div>
				</div>
			</div>
		</div>
	</div>

	<!-- Messages area (like chat) -->
	<div bind:this={messagesContainer} class="flex-1 overflow-y-auto">
		<div class="max-w-5xl mx-auto px-4 py-8">
			{#if loading}
				<!-- Loading state -->
				<div class="flex items-center justify-center h-[60vh]">
					<div class="text-center">
						<Spinner className="w-8 h-8 mx-auto mb-3" />
						<p class="text-gray-500 dark:text-gray-400">{$i18n.t('Loading analysis history...')}</p>
					</div>
				</div>
			{:else if error && !reportContent}
				<!-- Error state -->
				<div class="flex items-center justify-center h-[60vh]">
					<div class="text-center max-w-md">
						<div
							class="w-12 h-12 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center"
						>
							<svg
								class="w-6 h-6 text-red-500"
								fill="none"
								stroke="currentColor"
								viewBox="0 0 24 24"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
									d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
								/>
							</svg>
						</div>
						<p class="text-red-600 dark:text-red-400">{error}</p>
						<button
							class="mt-4 px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
							on:click={loadHistory}
						>
							{$i18n.t('Retry')}
						</button>
					</div>
				</div>
			{:else if !reportContent && !generatingReport && historyRecords.length === 0}
				<!-- Empty state - no analysis data -->
				<div class="flex items-center justify-center h-[60vh]">
					<div class="text-center max-w-md">
						<div
							class="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center"
						>
							<svg
								class="w-8 h-8 text-gray-400"
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
						</div>
						<h3 class="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
							{$i18n.t('No Analysis Data')}
						</h3>
						<p class="text-gray-500 dark:text-gray-400 mb-4">
							{$i18n.t('Run some reasoning analyses first to generate a report')}
						</p>
						<a
							href="/"
							class="inline-flex items-center gap-2 px-4 py-2 text-sm text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg"
						>
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
									d="M12 6v6m0 0v6m0-6h6m-6 0H6"
								/>
							</svg>
							{$i18n.t('Start a new chat')}
						</a>
					</div>
				</div>
			{:else if !reportContent && !generatingReport && eligibleReasoningModels.length === 0}
				<!-- No eligible models - all have <= 3 analyses -->
				<div class="flex items-center justify-center h-[60vh]">
					<div class="text-center max-w-md">
						<div
							class="w-16 h-16 mx-auto mb-4 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center"
						>
							<svg
								class="w-8 h-8 text-amber-500"
								fill="none"
								stroke="currentColor"
								viewBox="0 0 24 24"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="1.5"
									d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
								/>
							</svg>
						</div>
						<h3 class="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
							{$i18n.t('Insufficient Analysis Data')}
						</h3>
						<p class="text-gray-500 dark:text-gray-400 mb-4">
							{$i18n.t('Each reasoning model needs at least')}
							<span class="font-semibold text-amber-600 dark:text-amber-400"
								>{MIN_ANALYSES_FOR_REPORT}</span
							>
							{$i18n.t('analysis records to generate a report')}.
						</p>
						<p class="text-sm text-gray-400 dark:text-gray-500 mb-4">
							{$i18n.t('Current models')}:
							{#each Object.entries(modelAnalysisCounts) as [model, count], i}
								<span class="inline-flex items-center gap-1 mx-1">
									<span class="font-medium">{model}</span>
									<span class="text-xs px-1.5 py-0.5 rounded-full bg-gray-100 dark:bg-gray-800"
										>{count}</span
									>
									{#if i < Object.keys(modelAnalysisCounts).length - 1},{/if}
								</span>
							{/each}
						</p>
						<a
							href="/"
							class="inline-flex items-center gap-2 px-4 py-2 text-sm text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg"
						>
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
									d="M12 6v6m0 0v6m0-6h6m-6 0H6"
								/>
							</svg>
							{$i18n.t('Run more analyses')}
						</a>
					</div>
				</div>
			{:else if !reportContent && !generatingReport}
				<!-- Ready state - prompt to generate with saved reports -->
				<div class="space-y-8">
					<!-- Generation section -->
					<div class="flex items-center justify-center">
						<div class="text-center max-w-lg">
							<div
								class="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 flex items-center justify-center"
							>
								<svg
									class="w-10 h-10 text-blue-600 dark:text-blue-400"
									fill="none"
									stroke="currentColor"
									viewBox="0 0 24 24"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										stroke-width="1.5"
										d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
									/>
								</svg>
							</div>
							<h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-3">
								{$i18n.t('Ready to Generate Report')}
							</h2>
							<p class="text-gray-500 dark:text-gray-400 mb-6">
								{$i18n.t(
									'Select a reasoning model and a report generation model above to analyze its reasoning patterns'
								)}.
							</p>

							<!-- Eligible models info -->
							<div class="mb-6 text-sm">
								<p class="text-gray-600 dark:text-gray-400 mb-2">
									{$i18n.t('Available models for analysis')}:
								</p>
								<div class="flex flex-wrap justify-center gap-2">
									{#each eligibleReasoningModels as { model, count }}
										<span
											class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
										>
											<span class="font-medium">{model}</span>
											<span class="text-xs opacity-70">({count})</span>
										</span>
									{/each}
								</div>
							</div>

							<button
								class="inline-flex items-center gap-2 px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-500/25"
								on:click={handleGenerateReport}
								disabled={generatingReport || !selectedReportModel || !selectedReasoningModel}
							>
								<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										stroke-width="2"
										d="M13 10V3L4 14h7v7l9-11h-7z"
									/>
								</svg>
								{$i18n.t('Generate Report')}
							</button>

							{#if !selectedReasoningModel && !selectedReportModel}
								<p class="mt-3 text-xs text-gray-400 dark:text-gray-500">
									{$i18n.t('Please select both a reasoning model and a report model')}
								</p>
							{:else if !selectedReasoningModel}
								<p class="mt-3 text-xs text-gray-400 dark:text-gray-500">
									{$i18n.t('Please select a reasoning model to analyze')}
								</p>
							{:else if !selectedReportModel}
								<p class="mt-3 text-xs text-gray-400 dark:text-gray-500">
									{$i18n.t('Please select a report model')}
								</p>
							{/if}
						</div>
					</div>

					<!-- Saved Reports Section -->
					{#if savedReports.length > 0}
						<div class="border-t border-gray-100 dark:border-gray-800 pt-8">
							<div class="flex items-center justify-between mb-4">
								<h3
									class="text-lg font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2"
								>
									<svg
										class="w-5 h-5 text-gray-500"
										fill="none"
										stroke="currentColor"
										viewBox="0 0 24 24"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											stroke-width="2"
											d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"
										/>
									</svg>
									{$i18n.t('Saved Reports')}
									<span class="text-sm font-normal text-gray-500">({savedReports.length})</span>
								</h3>
								{#if loadingSavedReports}
									<Spinner className="w-4 h-4" />
								{/if}
							</div>

							<div class="grid gap-3">
								{#each savedReports as report (report.report_id)}
									<div
										class="group relative p-4 bg-white dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-xl hover:border-blue-300 dark:hover:border-blue-700 hover:shadow-md transition-all cursor-pointer"
										on:click={() => handleLoadSavedReport(report.report_id)}
										on:keydown={(e) => e.key === 'Enter' && handleLoadSavedReport(report.report_id)}
										role="button"
										tabindex="0"
									>
										<div class="flex items-start justify-between gap-4">
											<div class="flex-1 min-w-0">
												<div class="flex items-center gap-2 mb-2">
													<span class="text-sm font-medium text-gray-900 dark:text-gray-100">
														{report.reasoning_model}
													</span>
													<span
														class="text-xs px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
													>
														{report.records_count}
														{$i18n.t('analyses')}
													</span>
													<span
														class="text-xs px-2 py-0.5 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
													>
														{report.language === 'zh' ? '中文' : 'EN'}
													</span>
												</div>
												<p class="text-sm text-gray-500 dark:text-gray-400 line-clamp-2">
													{report.content_preview}
												</p>
												<div
													class="flex items-center gap-4 mt-2 text-xs text-gray-400 dark:text-gray-500"
												>
													<span class="flex items-center gap-1">
														<svg
															class="w-3.5 h-3.5"
															fill="none"
															stroke="currentColor"
															viewBox="0 0 24 24"
														>
															<path
																stroke-linecap="round"
																stroke-linejoin="round"
																stroke-width="2"
																d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
															/>
														</svg>
														{formatTimestamp(report.timestamp)}
													</span>
													<span class="flex items-center gap-1">
														<svg
															class="w-3.5 h-3.5"
															fill="none"
															stroke="currentColor"
															viewBox="0 0 24 24"
														>
															<path
																stroke-linecap="round"
																stroke-linejoin="round"
																stroke-width="2"
																d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
															/>
														</svg>
														{report.report_model}
													</span>
												</div>
											</div>

											<!-- Delete button -->
											<button
												class="opacity-0 group-hover:opacity-100 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-all"
												on:click|stopPropagation={() => handleDeleteSavedReport(report.report_id)}
												title={$i18n.t('Delete')}
											>
												<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
													<path
														stroke-linecap="round"
														stroke-linejoin="round"
														stroke-width="2"
														d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
													/>
												</svg>
											</button>
										</div>
									</div>
								{/each}
							</div>
						</div>
					{/if}
				</div>
			{:else}
				<!-- Report content (like a chat message) -->
				<div transition:fade={{ duration: 150 }}>
					<!-- Assistant message bubble -->
					<div class="flex gap-4">
						<!-- Avatar -->
						<div class="flex-shrink-0">
							<div
								class="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center"
							>
								<svg
									class="w-5 h-5 text-white"
									fill="none"
									stroke="currentColor"
									viewBox="0 0 24 24"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										stroke-width="2"
										d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
									/>
								</svg>
							</div>
						</div>

						<!-- Message content -->
						<div class="flex-1 min-w-0">
							<div class="flex items-center gap-2 mb-3">
								<span class="font-semibold text-gray-900 dark:text-gray-100">
									{$i18n.t('Analysis Report')}
								</span>
								<span
									class="text-xs px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
								>
									{selectedReasoningModel}
								</span>
								{#if loadedReportId}
									<span
										class="text-xs px-2 py-0.5 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
									>
										{$i18n.t('Loaded')}
									</span>
								{/if}
							</div>

							<!-- Content area -->
							{#if generatingReport && !displayContent}
								<!-- Skeleton loading when waiting for first content -->
								<div class="flex items-center gap-1">
									<Skeleton />
								</div>
							{:else if generatingReport && displayContent}
								<!-- During streaming: show plain text for performance -->
								<div class="prose prose-sm dark:prose-invert max-w-none streaming-text">
									<pre
										class="whitespace-pre-wrap font-sans text-sm text-gray-800 dark:text-gray-200 bg-transparent p-0 m-0 overflow-visible">{displayContent}</pre>
									<!-- Typing cursor inline when streaming -->
									<span
										class="inline-block w-1.5 h-4 ml-0.5 -mb-0.5 bg-gray-400 dark:bg-gray-500 animate-pulse rounded-sm"
									></span>
								</div>
							{:else if displayContent}
								<!-- After completion: render full Markdown -->
								<div class="prose prose-sm dark:prose-invert max-w-none">
									<Markdown id="report-content" content={displayContent} done={true} />
								</div>
							{/if}

							<!-- Progress indicator -->
							{#if generatingReport && progressMessage}
								<div
									class="flex items-center gap-2 mt-4 text-sm text-gray-500 dark:text-gray-400"
									transition:fade
								>
									<Spinner className="w-3 h-3" />
									<span>{progressMessage}</span>
								</div>
							{/if}

							<!-- Error display during/after generation -->
							{#if error && (reportContent || generatingReport)}
								<div
									class="mt-4 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
								>
									<div class="flex items-start gap-2">
										<svg
											class="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5"
											fill="none"
											stroke="currentColor"
											viewBox="0 0 24 24"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
											/>
										</svg>
										<div class="flex-1">
											<p class="text-sm text-red-700 dark:text-red-300">{error}</p>
											<button
												class="mt-2 text-xs text-red-600 dark:text-red-400 hover:underline"
												on:click={handleGenerateReport}
											>
												{$i18n.t('Retry')}
											</button>
										</div>
									</div>
								</div>
							{/if}

							<!-- Action buttons after generation -->
							{#if reportContent && !generatingReport}
								<div
									class="flex items-center gap-3 mt-6 pt-4 border-t border-gray-100 dark:border-gray-800"
								>
									<button
										class="flex items-center gap-2 px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
										on:click={handleCopy}
									>
										<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
											/>
										</svg>
										{$i18n.t('Copy')}
									</button>
									<button
										class="flex items-center gap-2 px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
										on:click={handleExport}
									>
										<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
											/>
										</svg>
										{$i18n.t('Export')}
									</button>
									<div class="flex-1"></div>
									<button
										class="flex items-center gap-2 px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
										on:click={() => {
											reportContent = '';
											displayContent = '';
											loadedReportId = null;
										}}
									>
										<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M10 19l-7-7m0 0l7-7m-7 7h18"
											/>
										</svg>
										{$i18n.t('Back')}
									</button>
								</div>
							{/if}
						</div>
					</div>
				</div>
			{/if}
		</div>
	</div>

	<!-- Bottom bar (only when generating or ready to generate again) -->
	{#if reportContent && !generatingReport}
		<div
			class="flex-shrink-0 border-t border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-3"
		>
			<div class="max-w-5xl mx-auto flex items-center justify-between">
				<span class="text-sm text-gray-500 dark:text-gray-400">
					{#if loadedReportId}
						{$i18n.t('Loaded saved report')}
					{:else}
						{$i18n.t('Report generated successfully')}
					{/if}
				</span>
				<div class="flex items-center gap-3">
					{#if loadedReportId}
						<button
							class="flex items-center gap-2 px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
							on:click={() => loadedReportId && handleDeleteSavedReport(loadedReportId)}
						>
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
									d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
								/>
							</svg>
							{$i18n.t('Delete')}
						</button>
					{/if}
					<button
						class="flex items-center gap-2 px-5 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all"
						on:click={handleGenerateReport}
						disabled={generatingReport || !selectedReportModel || !selectedReasoningModel}
					>
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
							/>
						</svg>
						{$i18n.t('Generate New Report')}
					</button>
				</div>
			</div>
		</div>
	{/if}

	{#if generatingReport}
		<div
			class="flex-shrink-0 border-t border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-3"
		>
			<div class="max-w-5xl mx-auto flex items-center justify-center gap-3">
				<Spinner className="w-4 h-4" />
				<span class="text-sm text-gray-500 dark:text-gray-400">
					{progressMessage || $i18n.t('Generating report...')}
				</span>
			</div>
		</div>
	{/if}
</div>

<style>
	/* Typing cursor animation */
	@keyframes blink {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0;
		}
	}

	/* Line clamp utility */
	.line-clamp-2 {
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
</style>
