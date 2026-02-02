/**
 * Analysis Tasks Store
 *
 * Manages multiple independent error analysis tasks that can run in parallel.
 * This allows users to start an analysis for one conversation and continue
 * using other conversations while the analysis runs.
 */

import { writable, derived, get } from 'svelte/store';
import { WEBUI_BASE_URL } from '$lib/constants';

/**
 * Status of an analysis task
 */
export type AnalysisTaskStatus =
	| 'pending' // Task created but not started
	| 'running' // Task is actively running
	| 'layer1' // Running Layer 1 analysis
	| 'layer2' // Running Layer 2 analysis
	| 'error_detection' // Running error detection
	| 'complete' // Task completed successfully
	| 'error' // Task failed with error
	| 'cancelled'; // Task was cancelled

/**
 * Overthinking analysis result
 */
export interface OverthinkingAnalysis {
	score: number;
	first_correct_answer_section: number | null;
	all_answer_sections: number[];
	total_sections: number;
}

/**
 * Error detection result
 */
export interface ErrorDetectionResult {
	errors: any[];
	overthinking_analysis?: OverthinkingAnalysis;
	claims?: any[];
	query_answers?: any[];
}

/**
 * Analysis result data
 */
export interface AnalysisResult {
	nodes: any[];
	edges: any[];
	sections: any[];
	analysis_metadata?: any;
}

/**
 * Represents a single analysis task
 */
export interface AnalysisTask {
	// Unique identifier for this task
	id: string;

	// Chat and message identifiers
	chatId: string;
	messageId: string;

	// Model used for analysis
	model: string;
	modelName?: string;

	// Task status
	status: AnalysisTaskStatus;

	// Progress information - detailed tracking for each stage
	progress: {
		stage: string;
		// Layer 2 progress
		layer2Completed: number;
		layer2Total: number;
		currentNodeId?: string; // Currently processing node ID
		currentNodeLabel?: string; // Currently processing node label
		// Error detection progress
		errorDetectionBatch: number;
		errorDetectionTotalBatches: number;
		// Overall progress
		percentComplete: number;
	};

	// Timing
	createdAt: Date;
	startedAt?: Date;
	completedAt?: Date;

	// Results
	result?: AnalysisResult;
	errorDetection?: ErrorDetectionResult;
	error?: string;

	// Message preview (for display in task list)
	messagePreview?: string;
	chatTitle?: string;

	// Abort controller for cancellation
	abortController?: AbortController;

	// Whether user has seen this completed task
	seen: boolean;
}

/**
 * Store state
 */
interface AnalysisStoreState {
	tasks: Map<string, AnalysisTask>;
	activeTaskIds: string[];
}

// Create the main store
function createAnalysisStore() {
	const { subscribe, set, update } = writable<AnalysisStoreState>({
		tasks: new Map(),
		activeTaskIds: []
	});

	return {
		subscribe,

		/**
		 * Create a new analysis task
		 */
		createTask: (params: {
			chatId: string;
			messageId: string;
			model: string;
			modelName?: string;
			messagePreview?: string;
			chatTitle?: string;
		}): string => {
			const taskId = `${params.chatId}-${params.messageId}-${Date.now()}`;

			const task: AnalysisTask = {
				id: taskId,
				chatId: params.chatId,
				messageId: params.messageId,
				model: params.model,
				modelName: params.modelName,
				messagePreview: params.messagePreview,
				chatTitle: params.chatTitle,
				status: 'pending',
				progress: {
					stage: 'pending',
					layer2Completed: 0,
					layer2Total: 0,
					currentNodeId: undefined,
					currentNodeLabel: undefined,
					errorDetectionBatch: 0,
					errorDetectionTotalBatches: 0,
					percentComplete: 0
				},
				createdAt: new Date(),
				seen: false
			};

			update((state) => {
				state.tasks.set(taskId, task);
				return state;
			});

			return taskId;
		},

		/**
		 * Update task status and progress
		 */
		updateTask: (taskId: string, updates: Partial<AnalysisTask>) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task) {
					state.tasks.set(taskId, { ...task, ...updates });
				}
				return state;
			});
		},

		/**
		 * Start a task
		 */
		startTask: (taskId: string, abortController?: AbortController) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task) {
					task.status = 'running';
					task.startedAt = new Date();
					task.abortController = abortController;
					if (!state.activeTaskIds.includes(taskId)) {
						state.activeTaskIds.push(taskId);
					}
				}
				return state;
			});
		},

		/**
		 * Update task progress
		 */
		updateProgress: (taskId: string, progress: Partial<AnalysisTask['progress']>) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task) {
					task.progress = { ...task.progress, ...progress };
				}
				return state;
			});
		},

		/**
		 * Set task status
		 */
		setStatus: (taskId: string, status: AnalysisTaskStatus) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task) {
					task.status = status;
					if (status === 'complete' || status === 'error' || status === 'cancelled') {
						task.completedAt = new Date();
						state.activeTaskIds = state.activeTaskIds.filter((id) => id !== taskId);
					}
				}
				return state;
			});
		},

		/**
		 * Complete a task with results
		 */
		completeTask: (
			taskId: string,
			result: AnalysisResult,
			errorDetection?: ErrorDetectionResult
		) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task) {
					task.status = 'complete';
					task.completedAt = new Date();
					task.result = result;
					task.errorDetection = errorDetection;
					task.progress.percentComplete = 100;
					task.progress.stage = 'complete';
					state.activeTaskIds = state.activeTaskIds.filter((id) => id !== taskId);
				}
				return state;
			});
		},

		/**
		 * Fail a task with error
		 */
		failTask: (taskId: string, error: string) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task) {
					task.status = 'error';
					task.completedAt = new Date();
					task.error = error;
					state.activeTaskIds = state.activeTaskIds.filter((id) => id !== taskId);
				}
				return state;
			});
		},

		/**
		 * Cancel a task
		 */
		cancelTask: (taskId: string) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task) {
					if (task.abortController) {
						task.abortController.abort();
					}
					task.status = 'cancelled';
					task.completedAt = new Date();
					state.activeTaskIds = state.activeTaskIds.filter((id) => id !== taskId);
				}
				return state;
			});
		},

		/**
		 * Mark a task as seen
		 */
		markSeen: (taskId: string) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task) {
					task.seen = true;
				}
				return state;
			});
		},

		/**
		 * Remove a task
		 */
		removeTask: (taskId: string) => {
			update((state) => {
				const task = state.tasks.get(taskId);
				if (task?.abortController) {
					task.abortController.abort();
				}
				state.tasks.delete(taskId);
				state.activeTaskIds = state.activeTaskIds.filter((id) => id !== taskId);
				return state;
			});
		},

		/**
		 * Get a task by ID
		 */
		getTask: (taskId: string): AnalysisTask | undefined => {
			const state = get({ subscribe });
			return state.tasks.get(taskId);
		},

		/**
		 * Get all tasks for a specific message
		 */
		getTasksForMessage: (messageId: string): AnalysisTask[] => {
			const state = get({ subscribe });
			return Array.from(state.tasks.values()).filter((t) => t.messageId === messageId);
		},

		/**
		 * Clear all completed/cancelled tasks
		 */
		clearCompleted: () => {
			update((state) => {
				for (const [taskId, task] of state.tasks.entries()) {
					if (
						task.status === 'complete' ||
						task.status === 'error' ||
						task.status === 'cancelled'
					) {
						state.tasks.delete(taskId);
					}
				}
				return state;
			});
		},

		/**
		 * Reset the store
		 */
		reset: () => {
			update((state) => {
				for (const task of state.tasks.values()) {
					if (task.abortController) {
						task.abortController.abort();
					}
				}
				return {
					tasks: new Map(),
					activeTaskIds: []
				};
			});
		}
	};
}

// Export the store singleton
export const analysisStore = createAnalysisStore();

// Derived stores for common queries

/**
 * All analysis tasks as an array (sorted by creation time, newest first)
 */
export const analysisTasks = derived(analysisStore, ($store) =>
	Array.from($store.tasks.values()).sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
);

/**
 * Currently running tasks (includes pending tasks that are about to run)
 */
export const runningTasks = derived(analysisStore, ($store) =>
	Array.from($store.tasks.values()).filter(
		(t) =>
			t.status === 'pending' ||
			t.status === 'running' ||
			t.status === 'layer1' ||
			t.status === 'layer2' ||
			t.status === 'error_detection'
	)
);

/**
 * Get the most relevant task for a specific message
 * Prioritizes active tasks, then most recent completed task
 */
export function getTaskForMessage(messageId: string): AnalysisTask | null {
	const state = get(analysisStore);
	const tasks = Array.from(state.tasks.values()).filter((t) => t.messageId === messageId);

	// Find active task first (including pending - task has been initiated)
	const activeTask = tasks.find(
		(t) =>
			t.status === 'pending' ||
			t.status === 'running' ||
			t.status === 'layer1' ||
			t.status === 'layer2' ||
			t.status === 'error_detection'
	);
	if (activeTask) return activeTask;

	// Find most recent completed task
	const completedTasks = tasks
		.filter((t) => t.status === 'complete')
		.sort((a, b) => (b.completedAt?.getTime() ?? 0) - (a.completedAt?.getTime() ?? 0));

	return completedTasks[0] || null;
}

// Cache for derived message stores to avoid repeated derivations
const messageStoreCache = new Map<string, ReturnType<typeof derived>>();

// Result cache to avoid returning new objects when nothing changed
const resultCache = new Map<string, { key: string; value: any }>();

/**
 * Create a derived store for a specific message's analysis state
 * This is the primary way components should access analysis state
 * Uses caching to minimize recomputation and object creation
 */
export function createMessageAnalysisStore(messageId: string) {
	// Return cached store if available
	if (messageStoreCache.has(messageId)) {
		return messageStoreCache.get(messageId)!;
	}

	const store = derived(analysisStore, ($store) => {
		const tasks = Array.from($store.tasks.values()).filter((t) => t.messageId === messageId);

		// Find active task first (including pending - user already clicked the button)
		const activeTask = tasks.find(
			(t) =>
				t.status === 'pending' ||
				t.status === 'running' ||
				t.status === 'layer1' ||
				t.status === 'layer2' ||
				t.status === 'error_detection'
		);

		// Find most recent completed task
		const completedTask = tasks
			.filter((t) => t.status === 'complete')
			.sort((a, b) => (b.completedAt?.getTime() ?? 0) - (a.completedAt?.getTime() ?? 0))[0];

		const task = activeTask || completedTask || null;

		if (!task) {
			// Check cache for empty state
			const cached = resultCache.get(messageId);
			if (cached && cached.key === 'empty') {
				return cached.value;
			}
			const emptyResult = {
				hasTask: false,
				isLoading: false,
				stage: 'idle' as const,
				progress: {
					layer2Completed: 0,
					layer2Total: 0,
					currentNodeId: '',
					currentNodeLabel: '',
					errorDetectionBatch: 0,
					errorDetectionTotalBatches: 0,
					percentComplete: 0
				},
				result: null as AnalysisResult | null,
				errorDetection: null as ErrorDetectionResult | null,
				error: null as string | null,
				taskId: null as string | null
			};
			resultCache.set(messageId, { key: 'empty', value: emptyResult });
			return emptyResult;
		}

		const isLoading = activeTask !== undefined;
		const stage = (() => {
			switch (task.status) {
				case 'pending':
				case 'running':
				case 'layer1':
					return 'layer1';
				case 'layer2':
					return 'layer2';
				case 'error_detection':
					return 'error_detection';
				case 'complete':
					return 'complete';
				default:
					return 'idle';
			}
		})();

		// Create cache key based on task state
		const cacheKey = `${task.id}:${task.status}:${task.progress.percentComplete}:${task.result ? 'r' : ''}:${task.errorDetection ? 'e' : ''}`;
		const cached = resultCache.get(messageId);
		if (cached && cached.key === cacheKey) {
			return cached.value;
		}

		const result = {
			hasTask: true,
			isLoading,
			stage: stage as 'idle' | 'layer1' | 'layer2' | 'error_detection' | 'complete',
			progress: {
				layer2Completed: task.progress.layer2Completed,
				layer2Total: task.progress.layer2Total,
				currentNodeId: task.progress.currentNodeId || '',
				currentNodeLabel: task.progress.currentNodeLabel || '',
				errorDetectionBatch: task.progress.errorDetectionBatch,
				errorDetectionTotalBatches: task.progress.errorDetectionTotalBatches,
				percentComplete: task.progress.percentComplete
			},
			result: task.result || null,
			errorDetection: task.errorDetection || null,
			error: task.error || null,
			taskId: task.id
		};

		resultCache.set(messageId, { key: cacheKey, value: result });
		return result;
	});

	messageStoreCache.set(messageId, store);
	return store;
}

/**
 * Currently selected task ID for viewing details
 */
export const selectedAnalysisTaskId = writable<string | null>(null);

/**
 * Run an analysis task in the background.
 * This function is independent of any Svelte component and will continue
 * running even if the user navigates away.
 *
 * @param taskId - The task ID returned from createTask
 * @param token - The authentication token
 */
export async function runAnalysisTask(taskId: string, token: string): Promise<void> {
	const task = analysisStore.getTask(taskId);
	if (!task) {
		console.error('Task not found:', taskId);
		return;
	}

	const abortController = new AbortController();
	analysisStore.startTask(taskId, abortController);

	try {
		const url = `${WEBUI_BASE_URL}/api/v1/analysis/reasoning/stream`;

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				Authorization: `Bearer ${token}`
			},
			body: JSON.stringify({
				chat_id: task.chatId,
				message_id: task.messageId,
				model: task.model,
				force: false
			}),
			signal: abortController.signal
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(errorText || `HTTP ${response.status}`);
		}

		const reader = response.body?.getReader();
		if (!reader) {
			throw new Error('Response body is not readable');
		}

		const decoder = new TextDecoder();
		let buffer = '';
		let eventType = '';
		let eventData = '';
		let result: AnalysisResult | null = null;
		let errorDetection: ErrorDetectionResult | null = null;

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() || '';

			for (const line of lines) {
				if (line.startsWith('event:')) {
					eventType = line.slice(6).trim();
				} else if (line.startsWith('data:')) {
					eventData = line.slice(5).trim();

					if (eventData && eventType) {
						try {
							const data = JSON.parse(eventData);

							switch (eventType) {
								case 'layer1':
									analysisStore.setStatus(taskId, 'layer1');
									analysisStore.updateProgress(taskId, {
										stage: 'layer1',
										layer2Total: data.nodes?.length || 0,
										percentComplete: 30
									});
									result = {
										nodes: data.nodes || [],
										edges: data.edges || [],
										sections: data.sections || []
									};
									analysisStore.updateTask(taskId, { result });
									break;

								// Backend sends 'layer2_node' event for each node processed
								case 'layer2_node':
									analysisStore.setStatus(taskId, 'layer2');
									const completed = data.completed || 0;
									const total = data.total || 1;
									// Find the current node being processed
									const currentNode = result?.nodes?.find((n: any) => n.id === data.node_id);
									analysisStore.updateProgress(taskId, {
										stage: 'layer2',
										layer2Completed: completed,
										layer2Total: total,
										currentNodeId: data.node_id,
										currentNodeLabel: currentNode?.label || data.node_id,
										percentComplete: 30 + (completed / total) * 40
									});

									// Update node with layer2 data
									if (result && data.node_id) {
										const updatedNodes = result.nodes.map((node: any) => {
											if (node.id === data.node_id) {
												return {
													...node,
													layer2: data.layer2
												};
											}
											return node;
										});
										result = { ...result, nodes: updatedNodes };
										analysisStore.updateTask(taskId, { result });
									}
									break;

								// Error detection result (final)
								case 'error_detection':
									analysisStore.setStatus(taskId, 'error_detection');
									analysisStore.updateProgress(taskId, {
										stage: 'error_detection',
										percentComplete: 90
									});
									errorDetection = {
										errors: data.errors || [],
										overthinking_analysis: data.overthinking_analysis || null,
										claims: data.claims || [],
										query_answers: data.query_answers || []
									};
									analysisStore.updateTask(taskId, { errorDetection });
									break;

								case 'complete':
									result = {
										nodes: data.nodes || result?.nodes || [],
										edges: data.edges || result?.edges || [],
										sections: data.sections || result?.sections || [],
										analysis_metadata: data.analysis_metadata
									};
									analysisStore.completeTask(taskId, result, errorDetection || undefined);
									break;

								case 'cached':
									// Cached result coming next
									break;

								case 'error':
									throw new Error(data.error || 'Unknown streaming error');
							}
						} catch (parseError) {
							if (eventType === 'error') {
								throw parseError;
							}
							console.warn('Failed to parse event data:', parseError);
						}
					}
					eventType = '';
					eventData = '';
				}
			}
		}

		// If we got here without completing, complete now
		const currentTask = analysisStore.getTask(taskId);
		if (currentTask && currentTask.status !== 'complete') {
			if (result) {
				analysisStore.completeTask(taskId, result, errorDetection || undefined);
			} else {
				analysisStore.failTask(taskId, 'No result received');
			}
		}
	} catch (error: any) {
		if (error.name === 'AbortError') {
			// Task was cancelled, don't report as error
			return;
		}

		analysisStore.failTask(taskId, error.message || 'Analysis failed');
	}
}
