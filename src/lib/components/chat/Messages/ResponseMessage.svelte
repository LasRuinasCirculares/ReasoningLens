<script context="module" lang="ts">
	// Module-level code preserved for future use
</script>

<script lang="ts">
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import { createEventDispatcher, onDestroy } from 'svelte';
	import { onMount, tick, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType, t } from 'i18next';

	const i18n = getContext<Writable<i18nType>>('i18n');

	const dispatch = createEventDispatcher();

	import { createNewFeedback, getFeedbackById, updateFeedbackById } from '$lib/apis/evaluations';
	import { getChatById } from '$lib/apis/chats';
	import { generateTags } from '$lib/apis';

	import {
		audioQueue,
		config,
		models,
		settings,
		mobile,
		temporaryChatEnabled,
		TTSWorker,
		user,
		showSidebar,
		analysisStore,
		selectedAnalysisTaskId,
		runAnalysisTask,
		createMessageAnalysisStore
	} from '$lib/stores';
	import { synthesizeOpenAISpeech } from '$lib/apis/audio';
	import { imageGenerations } from '$lib/apis/images';
	import {
		copyToClipboard as _copyToClipboard,
		approximateToHumanReadable,
		getMessageContentParts,
		sanitizeResponseContent,
		createMessagesList,
		formatDate,
		removeDetails,
		removeAllDetails
	} from '$lib/utils';
	import { WEBUI_API_BASE_URL, WEBUI_BASE_URL } from '$lib/constants';

	import Name from './Name.svelte';
	import ProfileImage from './ProfileImage.svelte';
	import Skeleton from './Skeleton.svelte';
	import Image from '$lib/components/common/Image.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import RateComment from './RateComment.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import WebSearchResults from './ResponseMessage/WebSearchResults.svelte';
	import Sparkles from '$lib/components/icons/Sparkles.svelte';

	import DeleteConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';

	import ErrorMessage from './Error.svelte';
	import Citations from './Citations.svelte';
	import CodeExecutions from './CodeExecutions.svelte';
	import ContentRenderer from './ContentRenderer.svelte';
	import { KokoroWorker } from '$lib/workers/KokoroWorker';
	import FileItem from '$lib/components/common/FileItem.svelte';
	import FollowUps from './ResponseMessage/FollowUps.svelte';
	import { fade } from 'svelte/transition';
	import { flyAndScale } from '$lib/utils/transitions';
	import RegenerateMenu from './ResponseMessage/RegenerateMenu.svelte';
	import StatusHistory from './ResponseMessage/StatusHistory.svelte';
	import FullHeightIframe from '$lib/components/common/FullHeightIframe.svelte';
	import ReasoningTree from './ResponseMessage/ReasoningTree.svelte';
	import Selector from '$lib/components/chat/ModelSelector/Selector.svelte';
	import { SvelteFlowProvider } from '@xyflow/svelte';

	interface MessageType {
		id: string;
		model: string;
		content: string;
		files?: { type: string; url: string }[];
		timestamp: number;
		role: string;
		statusHistory?: {
			done: boolean;
			action: string;
			description: string;
			urls?: string[];
			query?: string;
		}[];
		status?: {
			done: boolean;
			action: string;
			description: string;
			urls?: string[];
			query?: string;
		};
		done: boolean;
		error?: boolean | { content: string };
		sources?: string[];
		code_executions?: {
			uuid: string;
			name: string;
			code: string;
			language?: string;
			result?: {
				error?: string;
				output?: string;
				files?: { name: string; url: string }[];
			};
		}[];
		info?: {
			openai?: boolean;
			prompt_tokens?: number;
			completion_tokens?: number;
			total_tokens?: number;
			eval_count?: number;
			eval_duration?: number;
			prompt_eval_count?: number;
			prompt_eval_duration?: number;
			total_duration?: number;
			load_duration?: number;
			usage?: unknown;
		};
		annotation?: { type: string; rating: number };
	}

	export let chatId = '';
	export let history;
	export let messageId;
	export let selectedModels = [];

	// Performance optimization: Use shallow comparison for message updates instead of deep JSON comparison
	// Only deep copy when critical fields actually change
	let message: MessageType = JSON.parse(JSON.stringify(history.messages[messageId]));
	let lastMessageId = messageId;
	let lastContentHash = message.content?.length ?? 0;
	let lastDoneState = message.done;

	$: if (history.messages && history.messages[messageId]) {
		const sourceMsg = history.messages[messageId];
		const contentChanged = (sourceMsg.content?.length ?? 0) !== lastContentHash;
		const doneChanged = sourceMsg.done !== lastDoneState;
		const idChanged = messageId !== lastMessageId;

		// Only deep copy when meaningful changes occur
		if (idChanged || contentChanged || doneChanged || sourceMsg.error !== message.error) {
			message = JSON.parse(JSON.stringify(sourceMsg));
			lastMessageId = messageId;
			lastContentHash = message.content?.length ?? 0;
			lastDoneState = message.done;
		}
	}

	export let siblings;

	export let setInputText: Function = () => {};
	export let gotoMessage: Function = () => {};
	export let showPreviousMessage: Function;
	export let showNextMessage: Function;

	export let updateChat: Function;
	export let editMessage: Function;
	export let saveMessage: Function;
	export let rateMessage: Function;
	export let actionMessage: Function;
	export let deleteMessage: Function;

	export let submitMessage: Function;
	export let continueResponse: Function;
	export let regenerateResponse: Function;

	export let addMessages: Function;

	export let isLastMessage = true;
	export let readOnly = false;
	export let editCodeBlock = true;
	export let topPadding = false;

	let citationsElement: HTMLDivElement;

	let contentContainerElement: HTMLDivElement;
	let buttonsContainerElement: HTMLDivElement;
	let showDeleteConfirm = false;

	let model = null;
	$: model = $models.find((m) => m.id === message.model);

	let edit = false;
	let editedContent = '';
	let editTextAreaElement: HTMLTextAreaElement;

	let messageIndexEdit = false;

	let speaking = false;
	let speakingIdx: number | undefined;

	let loadingSpeech = false;
	let generatingImage = false;

	let showRateComment = false;
	let showAnalysisPanel = false;
	// UI-only state - analysis data is now read from store
	// Use last selected analysis model from localStorage if available, otherwise use message model
	let analysisModelId =
		(typeof localStorage !== 'undefined' && localStorage.getItem('lastAnalysisModelId')) ||
		message.model;
	let analysisModel = null;
	let analysisMessageId = message.id;
	// Panel UI state
	const analysisCollapsedPeekPx = 56;
	const analysisInsetVarName = '--analysis-panel-inset';
	const analysisPanelMinWidth = 360;
	const analysisPanelMaxWidth = 1000;
	const analysisPanelMaxViewportRatio = 0.9;
	const analysisPanelDefaultWidth = 860;
	let analysisPanelWidth = analysisPanelDefaultWidth;
	let analysisCollapsed = false;
	let analysisInset = 0;
	let resizingAnalysis = false;
	let resizeStartX = 0;
	let resizeStartWidth = analysisPanelWidth;

	// Create a derived store for this message's analysis state
	// This is the single source of truth for analysis data
	const messageAnalysis = createMessageAnalysisStore(message.id);

	// For backward compatibility - these are now derived from the store
	$: analysisLoading = $messageAnalysis.isLoading;
	$: analysisError = $messageAnalysis.error || '';
	$: analysisResult = $messageAnalysis.result;
	$: errorDetectionResult = $messageAnalysis.errorDetection;
	$: analysisStage = $messageAnalysis.stage;

	// Auto-open analysis panel when analysis is in progress
	$: if (analysisLoading && !showAnalysisPanel && !$mobile) {
		showAnalysisPanel = true;
		analysisPanelWidth = analysisPanelDefaultWidth;
	}

	$: analysisLayer2Progress = {
		completed: $messageAnalysis.progress.layer2Completed,
		total: $messageAnalysis.progress.layer2Total,
		currentNodeId: $messageAnalysis.progress.currentNodeId,
		currentNodeLabel: $messageAnalysis.progress.currentNodeLabel
	};
	$: analysisErrorDetectionProgress = {
		batch: $messageAnalysis.progress.errorDetectionBatch,
		totalBatches: $messageAnalysis.progress.errorDetectionTotalBatches
	};
	$: backgroundTaskId = $messageAnalysis.taskId;

	// Pre-build highlight cache when analysis result becomes available (Layer 1 complete)
	// This ensures highlighting works immediately when user hovers a node
	$: if (analysisResult?.sections?.length > 0 && contentContainerElement) {
		// Use requestAnimationFrame to ensure DOM is rendered before building cache
		requestAnimationFrame(() => {
			if (contentContainerElement && analysisResult?.sections) {
				ensureHighlightCaches(analysisResult.sections);
			}
		});
	}

	// Debug logging disabled for production
	const logAnalysis = (_msg: string, _data: Record<string, unknown> = {}) => {
		// Uncomment for debugging: console.info(`[Analysis] ${_msg}`, _data);
	};

	const getDockedMaxWidth = () => {
		if (typeof window === 'undefined') return analysisPanelMaxWidth;
		return Math.min(
			analysisPanelMaxWidth,
			Math.round(window.innerWidth * analysisPanelMaxViewportRatio)
		);
	};

	const clampAnalysisWidth = (width: number) => {
		const maxWidth = getDockedMaxWidth();
		return Math.min(Math.max(width, analysisPanelMinWidth), maxWidth);
	};

	const startPanelResize = (event: PointerEvent) => {
		if ($mobile || analysisCollapsed) return;
		resizingAnalysis = true;
		resizeStartX = event.clientX;
		resizeStartWidth = analysisPanelWidth;

		window.addEventListener('pointermove', handlePanelResize);
		window.addEventListener('pointerup', stopPanelResize);
	};

	const handlePanelResize = (event: PointerEvent) => {
		if (!resizingAnalysis) return;
		const delta = resizeStartX - event.clientX;
		analysisPanelWidth = clampAnalysisWidth(resizeStartWidth + delta);
	};

	const stopPanelResize = () => {
		if (!resizingAnalysis) return;
		resizingAnalysis = false;
		window.removeEventListener('pointermove', handlePanelResize);
		window.removeEventListener('pointerup', stopPanelResize);
	};

	$: {
		// Keep docked panel width from squeezing the main chat area
		analysisPanelWidth = clampAnalysisWidth(analysisPanelWidth);
	}

	const handleWindowResize = () => {
		analysisPanelWidth = clampAnalysisWidth(analysisPanelWidth);
	};

	const analysisPanelOpenHandler = (event: CustomEvent) => {
		const openedId = event?.detail;
		if (openedId !== message.id && showAnalysisPanel) {
			closeAnalysisPanel();
		}
	};

	$: if (message?.id && message.id !== analysisMessageId) {
		analysisMessageId = message.id;
		// Keep the last used analysis model from localStorage, don't reset to message.model
		// This ensures users can consistently use their preferred analysis model
	}

	$: analysisModel =
		$models.find((m) => m.id === analysisModelId) ??
		$models.find((m) => m.id === message.model) ??
		null;

	// Validate analysisModelId exists in available models
	$: if (
		$models.length > 0 &&
		analysisModelId &&
		!$models.map((m) => m.id).includes(analysisModelId)
	) {
		// If saved model is not available, fall back to first available model
		analysisModelId = $models[0]?.id ?? '';
	}

	$: analysisInset =
		showAnalysisPanel && !$mobile
			? analysisCollapsed
				? analysisCollapsedPeekPx
				: analysisPanelWidth
			: 0;

	// Shared text + progress helpers for the analysis loading state
	$: analysisPrimaryStatus =
		analysisStage === 'layer1'
			? $i18n.t('Analyzing reasoning structure...')
			: analysisStage === 'layer2'
				? $i18n.t('Analyzing details...')
				: analysisStage === 'error_detection'
					? $i18n.t('Detecting errors...')
					: $i18n.t('Initializing analysis...');

	$: analysisSecondaryStatus =
		analysisStage === 'layer2'
			? analysisLayer2Progress.currentNodeLabel
				? `${analysisLayer2Progress.completed}/${analysisLayer2Progress.total || '?'} ${$i18n.t('nodes')} - ${analysisLayer2Progress.currentNodeLabel}`
				: `${analysisLayer2Progress.completed}/${analysisLayer2Progress.total || '?'} ${$i18n.t('nodes')}`
			: analysisStage === 'error_detection'
				? analysisErrorDetectionProgress.totalBatches > 0
					? `${$i18n.t('Batch')} ${analysisErrorDetectionProgress.batch}/${analysisErrorDetectionProgress.totalBatches}`
					: $i18n.t('Analyzing sections for errors...')
				: $i18n.t('Identifying key reasoning steps');

	$: analysisLoadingPercent = (() => {
		if (analysisStage === 'layer1') return 30;
		if (analysisStage === 'layer2') {
			if (analysisLayer2Progress.total > 0) {
				return Math.min(
					70,
					Math.max(
						30,
						30 + Math.round((analysisLayer2Progress.completed / analysisLayer2Progress.total) * 40)
					)
				);
			}
			return 45;
		}
		if (analysisStage === 'error_detection') {
			if (analysisErrorDetectionProgress.totalBatches > 0) {
				return Math.min(
					95,
					Math.max(
						70,
						70 +
							Math.round(
								(analysisErrorDetectionProgress.batch /
									analysisErrorDetectionProgress.totalBatches) *
									25
							)
					)
				);
			}
			return 80;
		}
		if (analysisStage === 'complete') return 100;
		return 18;
	})();

	$: if (!showAnalysisPanel && analysisCollapsed) {
		analysisCollapsed = false;
	}

	$: if (typeof document !== 'undefined') {
		document.documentElement.style.setProperty(analysisInsetVarName, `${analysisInset}px`);
	}

	const copyToClipboard = async (text) => {
		text = removeAllDetails(text);

		if (($config?.ui?.response_watermark ?? '').trim() !== '') {
			text = `${text}\n\n${$config?.ui?.response_watermark}`;
		}

		const res = await _copyToClipboard(text, null, $settings?.copyFormatted ?? false);
		if (res) {
			toast.success($i18n.t('Copying to clipboard was successful!'));
		}
	};

	const stopAudio = () => {
		try {
			speechSynthesis.cancel();
			$audioQueue.stop();
		} catch {}

		if (speaking) {
			speaking = false;
			speakingIdx = undefined;
		}
	};

	/**
	 * Start analysis using the global store.
	 * All analysis now runs through the store for consistent state management.
	 */
	const startAnalysis = async () => {
		// Check if there's already an active task for this message
		if (analysisLoading) {
			logAnalysis('startAnalysis:already_running');
			return;
		}

		const modelId = analysisModelId || message.model;
		const modelInfo = $models.find((m) => m.id === modelId);

		// Save the selected analysis model to localStorage for future use
		if (typeof localStorage !== 'undefined') {
			localStorage.setItem('lastAnalysisModelId', modelId);
		}

		// Get chat title for display
		let chatTitleForDisplay = '';
		try {
			const chat = await getChatById(localStorage.token, chatId);
			chatTitleForDisplay = chat?.title || '';
		} catch (e) {
			// Ignore errors fetching chat title
		}

		// Create a new task in the store
		const taskId = analysisStore.createTask({
			chatId,
			messageId: message.id,
			model: modelId,
			modelName: modelInfo?.name || modelId,
			messagePreview: message.content?.slice(0, 100),
			chatTitle: chatTitleForDisplay || $i18n.t('Chat')
		});

		logAnalysis('startAnalysis:created_task', { taskId, model: modelId });

		// Start the analysis using the global store function
		runAnalysisTask(taskId, localStorage.token).catch((err) => {
			console.error('Analysis failed:', err);
		});
	};

	const handleAnalyzeReasoning = async () => {
		if (typeof window !== 'undefined') {
			window.dispatchEvent(new CustomEvent('analysis-panel-open', { detail: message.id }));
		}

		// Close sidebar if it's open to maximize space for analysis
		if ($showSidebar) {
			showSidebar.set(false);
		}

		analysisCollapsed = false;
		showAnalysisPanel = true;

		// Use default width when opening (not maximized)
		if (!$mobile) {
			analysisPanelWidth = analysisPanelDefaultWidth;
		}

		// Don't auto-start analysis - user will click the Analyze button in the panel
	};

	const toggleAnalysisCollapse = () => {
		if ($mobile) {
			closeAnalysisPanel();
			return;
		}

		analysisCollapsed = !analysisCollapsed;
	};

	const closeAnalysisPanel = () => {
		// Cancel current task if running
		if (backgroundTaskId) {
			analysisStore.cancelTask(backgroundTaskId);
		}
		showAnalysisPanel = false;
		analysisCollapsed = false;
		stopPanelResize();
	};

	const speak = async () => {
		if (!(message?.content ?? '').trim().length) {
			toast.info($i18n.t('No content to speak'));
			return;
		}

		speaking = true;
		const content = removeAllDetails(message.content);

		if ($config.audio.tts.engine === '') {
			let voices = [];
			const getVoicesLoop = setInterval(() => {
				voices = speechSynthesis.getVoices();
				if (voices.length > 0) {
					clearInterval(getVoicesLoop);

					const voice =
						voices
							?.filter(
								(v) => v.voiceURI === ($settings?.audio?.tts?.voice ?? $config?.audio?.tts?.voice)
							)
							?.at(0) ?? undefined;

					console.log(voice);

					const speech = new SpeechSynthesisUtterance(content);
					speech.rate = $settings.audio?.tts?.playbackRate ?? 1;

					console.log(speech);

					speech.onend = () => {
						speaking = false;
						if ($settings.conversationMode) {
							document.getElementById('voice-input-button')?.click();
						}
					};

					if (voice) {
						speech.voice = voice;
					}

					speechSynthesis.speak(speech);
				}
			}, 100);
		} else {
			$audioQueue.setId(`${message.id}`);
			$audioQueue.setPlaybackRate($settings.audio?.tts?.playbackRate ?? 1);
			$audioQueue.onStopped = () => {
				speaking = false;
				speakingIdx = undefined;
			};

			loadingSpeech = true;
			const messageContentParts: string[] = getMessageContentParts(
				content,
				$config?.audio?.tts?.split_on ?? 'punctuation'
			);

			if (!messageContentParts.length) {
				console.log('No content to speak');
				toast.info($i18n.t('No content to speak'));

				speaking = false;
				loadingSpeech = false;
				return;
			}

			console.debug('Prepared message content for TTS', messageContentParts);
			if ($settings.audio?.tts?.engine === 'browser-kokoro') {
				if (!$TTSWorker) {
					await TTSWorker.set(
						new KokoroWorker({
							dtype: $settings.audio?.tts?.engineConfig?.dtype ?? 'fp32'
						})
					);

					await $TTSWorker.init();
				}

				for (const [idx, sentence] of messageContentParts.entries()) {
					const url = await $TTSWorker
						.generate({
							text: sentence,
							voice: $settings?.audio?.tts?.voice ?? $config?.audio?.tts?.voice
						})
						.catch((error) => {
							console.error(error);
							toast.error(`${error}`);

							speaking = false;
							loadingSpeech = false;
						});

					if (url && speaking) {
						$audioQueue.enqueue(url);
						loadingSpeech = false;
					}
				}
			} else {
				for (const [idx, sentence] of messageContentParts.entries()) {
					const res = await synthesizeOpenAISpeech(
						localStorage.token,
						$settings?.audio?.tts?.defaultVoice === $config.audio.tts.voice
							? ($settings?.audio?.tts?.voice ?? $config?.audio?.tts?.voice)
							: $config?.audio?.tts?.voice,
						sentence
					).catch((error) => {
						console.error(error);
						toast.error(`${error}`);

						speaking = false;
						loadingSpeech = false;
					});

					if (res && speaking) {
						const blob = await res.blob();
						const url = URL.createObjectURL(blob);

						$audioQueue.enqueue(url);
						loadingSpeech = false;
					}
				}
			}
		}
	};

	let preprocessedDetailsCache = [];

	function preprocessForEditing(content: string): string {
		// Replace <details>...</details> with unique ID placeholder
		const detailsBlocks = [];
		let i = 0;

		content = content.replace(/<details[\s\S]*?<\/details>/gi, (match) => {
			detailsBlocks.push(match);
			return `<details id="__DETAIL_${i++}__"/>`;
		});

		// Store original blocks in the editedContent or globally (see merging later)
		preprocessedDetailsCache = detailsBlocks;

		return content;
	}

	function postprocessAfterEditing(content: string): string {
		const restoredContent = content.replace(
			/<details id="__DETAIL_(\d+)__"\/>/g,
			(_, index) => preprocessedDetailsCache[parseInt(index)] || ''
		);

		return restoredContent;
	}

	const editMessageHandler = async () => {
		edit = true;

		editedContent = preprocessForEditing(message.content);

		await tick();

		editTextAreaElement.style.height = '';
		editTextAreaElement.style.height = `${editTextAreaElement.scrollHeight}px`;
	};

	const editMessageConfirmHandler = async () => {
		const messageContent = postprocessAfterEditing(editedContent ? editedContent : '');
		editMessage(message.id, { content: messageContent }, false);

		edit = false;
		editedContent = '';

		await tick();
	};

	const saveAsCopyHandler = async () => {
		const messageContent = postprocessAfterEditing(editedContent ? editedContent : '');

		editMessage(message.id, { content: messageContent });

		edit = false;
		editedContent = '';

		await tick();
	};

	const cancelEditMessage = async () => {
		edit = false;
		editedContent = '';
		await tick();
	};

	const generateImage = async (message: MessageType) => {
		generatingImage = true;
		const res = await imageGenerations(localStorage.token, message.content).catch((error) => {
			toast.error(`${error}`);
		});
		console.log(res);

		if (res) {
			const files = res.map((image) => ({
				type: 'image',
				url: `${image.url}`
			}));

			saveMessage(message.id, {
				...message,
				files: files
			});
		}

		generatingImage = false;
	};

	let feedbackLoading = false;

	const feedbackHandler = async (rating: number | null = null, details: object | null = null) => {
		feedbackLoading = true;
		console.log('Feedback', rating, details);

		const updatedMessage = {
			...message,
			annotation: {
				...(message?.annotation ?? {}),
				...(rating !== null ? { rating: rating } : {}),
				...(details ? details : {})
			}
		};

		const chat = await getChatById(localStorage.token, chatId).catch((error) => {
			toast.error(`${error}`);
		});
		if (!chat) {
			return;
		}

		const messages = createMessagesList(history, message.id);

		let feedbackItem = {
			type: 'rating',
			data: {
				...(updatedMessage?.annotation ? updatedMessage.annotation : {}),
				model_id: message?.selectedModelId ?? message.model,
				...(history.messages[message.parentId].childrenIds.length > 1
					? {
							sibling_model_ids: history.messages[message.parentId].childrenIds
								.filter((id) => id !== message.id)
								.map((id) => history.messages[id]?.selectedModelId ?? history.messages[id].model)
						}
					: {})
			},
			meta: {
				arena: message ? message.arena : false,
				model_id: message.model,
				message_id: message.id,
				message_index: messages.length,
				chat_id: chatId
			},
			snapshot: {
				chat: chat
			}
		};

		const baseModels = [
			feedbackItem.data.model_id,
			...(feedbackItem.data.sibling_model_ids ?? [])
		].reduce((acc, modelId) => {
			const model = $models.find((m) => m.id === modelId);
			if (model) {
				acc[model.id] = model?.info?.base_model_id ?? null;
			} else {
				// Log or handle cases where corresponding model is not found
				console.warn(`Model with ID ${modelId} not found`);
			}
			return acc;
		}, {});
		feedbackItem.meta.base_models = baseModels;

		let feedback = null;
		if (message?.feedbackId) {
			feedback = await updateFeedbackById(
				localStorage.token,
				message.feedbackId,
				feedbackItem
			).catch((error) => {
				toast.error(`${error}`);
			});
		} else {
			feedback = await createNewFeedback(localStorage.token, feedbackItem).catch((error) => {
				toast.error(`${error}`);
			});

			if (feedback) {
				updatedMessage.feedbackId = feedback.id;
			}
		}

		console.log(updatedMessage);
		saveMessage(message.id, updatedMessage);

		await tick();

		if (!details) {
			showRateComment = true;

			if (!updatedMessage.annotation?.tags && (message?.content ?? '') !== '') {
				// attempt to generate tags
				const tags = await generateTags(localStorage.token, message.model, messages, chatId).catch(
					(error) => {
						console.error(error);
						return [];
					}
				);
				console.log(tags);

				if (tags) {
					updatedMessage.annotation.tags = tags;
					feedbackItem.data.tags = tags;

					saveMessage(message.id, updatedMessage);
					await updateFeedbackById(
						localStorage.token,
						updatedMessage.feedbackId,
						feedbackItem
					).catch((error) => {
						toast.error(`${error}`);
					});
				}
			}
		}

		feedbackLoading = false;
	};

	const deleteMessageHandler = async () => {
		deleteMessage(message.id);
	};

	$: if (!edit) {
		(async () => {
			await tick();
		})();
	}

	const buttonsWheelHandler = (event: WheelEvent) => {
		if (buttonsContainerElement) {
			if (buttonsContainerElement.scrollWidth <= buttonsContainerElement.clientWidth) {
				// If the container is not scrollable, horizontal scroll
				return;
			} else {
				event.preventDefault();

				if (event.deltaY !== 0) {
					// Adjust horizontal scroll position based on vertical scroll
					buttonsContainerElement.scrollLeft += event.deltaY;
				}
			}
		}
	};

	const contentCopyHandler = (e) => {
		if (contentContainerElement) {
			e.preventDefault();
			// Get the selected HTML
			const selection = window.getSelection();
			const range = selection.getRangeAt(0);
			const tempDiv = document.createElement('div');

			// Remove background, color, and font styles
			tempDiv.appendChild(range.cloneContents());

			tempDiv.querySelectorAll('table').forEach((table) => {
				table.style.borderCollapse = 'collapse';
				table.style.width = 'auto';
				table.style.tableLayout = 'auto';
			});

			tempDiv.querySelectorAll('th').forEach((th) => {
				th.style.whiteSpace = 'nowrap';
				th.style.padding = '4px 8px';
			});

			// Put cleaned HTML + plain text into clipboard
			e.clipboardData.setData('text/html', tempDiv.innerHTML);
			e.clipboardData.setData('text/plain', selection.toString());
		}
	};

	let reasoningHighlights: HTMLElement[] = [];

	// ========== Highlight Optimization: Pre-computed text node cache ==========
	// Cache structure for fast highlighting - built once when content is ready
	interface TextNodeCache {
		fullText: string;
		textNodes: { node: Text | null; length: number; startOffset: number }[];
		targetElement: Element;
		builtAt: number;
	}
	let textNodeCache: TextNodeCache | null = null;
	let textNodeCacheKey = ''; // Used to detect when content changes

	// Pre-computed section position cache: sectionNum -> { start, end } in fullText
	interface SectionPositionCache {
		[sectionNum: number]: { start: number; end: number };
	}
	let sectionPositionCache: SectionPositionCache = {};
	let sectionCacheKey = '';

	/**
	 * Expand the reasoning collapsible area if it's collapsed.
	 * This is needed because Collapsible only renders content when open.
	 * Returns true if the area is now expanded or was already expanded.
	 */
	const ensureReasoningExpanded = (): boolean => {
		if (!contentContainerElement) return false;

		const reasoningContainer = contentContainerElement.querySelector('div[data-type="reasoning"]');
		if (!reasoningContainer) return true; // No reasoning area, proceed with content container

		// Check if the content is already visible (Collapsible renders content in a transition div)
		// The content slot is wrapped in a div with transition:slide when open
		const contentDivs = reasoningContainer.querySelectorAll(':scope > div');
		let hasContent = false;

		for (const div of contentDivs) {
			// The content div doesn't have cursor-pointer class (that's the header)
			if (!div.classList.contains('cursor-pointer')) {
				// This is the content area - check if it has actual reasoning text
				const text = div.textContent?.trim() ?? '';
				if (text.length > 50) {
					// Has substantial content
					hasContent = true;
					break;
				}
			}
		}

		if (!hasContent) {
			// Need to expand - find and click the header
			const headerDiv = reasoningContainer.querySelector('div.cursor-pointer');
			if (headerDiv) {
				// Simulate a pointerup event to toggle the Collapsible
				headerDiv.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }));
				return false; // Content not yet rendered, will need retry
			}
		}

		return true; // Already expanded or no action needed
	};

	/**
	 * Build the text node cache for fast highlighting.
	 * This should be called once when content is loaded or changed.
	 */
	const buildTextNodeCache = (): TextNodeCache | null => {
		if (!contentContainerElement) return null;

		// Find reasoning container (rendered as div with data-type="reasoning" by Collapsible component)
		const reasoningContainers = Array.from(
			contentContainerElement.querySelectorAll('div[data-type="reasoning"]')
		);
		const targetEl = reasoningContainers.length ? reasoningContainers[0] : contentContainerElement;

		// Collapsible uses div, not details, so we don't need to force open
		// Just walk through text nodes

		const textNodes: { node: Text | null; length: number; startOffset: number }[] = [];
		let fullText = '';
		let currentOffset = 0;

		const walker = document.createTreeWalker(
			targetEl,
			NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT
		);
		while (walker.nextNode()) {
			const node = walker.currentNode;

			if (node.nodeType === Node.ELEMENT_NODE) {
				if ((node as Element).tagName === 'BR') {
					textNodes.push({ node: null, length: 1, startOffset: currentOffset });
					fullText += '\n';
					currentOffset += 1;
				}
				continue;
			}

			const textNode = node as Text;
			// Skip text nodes in the Collapsible header area
			// The header contains "Thought for X seconds" or "Thinking..." text
			// and is wrapped in a div with cursor-pointer class
			let parent = textNode.parentElement;
			let inHeader = false;
			while (parent && parent !== targetEl) {
				// Skip the header button div (has cursor-pointer class and is direct child of the container)
				if (parent.classList?.contains('cursor-pointer') && parent.parentElement === targetEl) {
					inHeader = true;
					break;
				}
				parent = parent.parentElement;
			}
			if (inHeader) continue;

			const value = textNode.textContent ?? '';
			if (!value) continue;

			textNodes.push({ node: textNode, length: value.length, startOffset: currentOffset });
			fullText += value;
			currentOffset += value.length;
		}

		return {
			fullText,
			textNodes,
			targetElement: targetEl,
			builtAt: Date.now()
		};
	};

	/**
	 * Build section position cache from sections array.
	 * Maps section numbers to their positions in the cached fullText.
	 *
	 * Strategy (optimized):
	 * 1. First try to use backend-provided start_pos/end_pos (if available and valid)
	 * 2. Fall back to text search if backend positions don't match
	 * 3. Use the START of the NEXT section as the END of the current section
	 *    (This avoids length mismatch issues due to whitespace normalization)
	 * 4. For the last section, use the end of the fullText
	 */
	const buildSectionPositionCache = (
		sections: { section_num: number; text: string; start_pos?: number; end_pos?: number }[],
		cache: TextNodeCache
	): SectionPositionCache => {
		if (!sections || !sections.length || !cache) return {};

		const positions: SectionPositionCache = {};

		// Sort sections by section_num to ensure sequential order
		const sortedSections = [...sections].sort((a, b) => a.section_num - b.section_num);

		// Quick check: if backend provides valid positions and they're within range, use them
		const backendPositionsValid = sortedSections.every(
			(s) =>
				typeof s.start_pos === 'number' &&
				typeof s.end_pos === 'number' &&
				s.start_pos >= 0 &&
				s.end_pos > s.start_pos
		);

		// Calculate offset: backend positions are relative to reasoning text start
		// We need to find where the reasoning content starts in our fullText
		let positionOffset = 0;
		if (backendPositionsValid && sortedSections.length > 0) {
			const firstSection = sortedSections[0];
			const firstSectionText = normalizeHighlightText(firstSection.text);
			if (firstSectionText) {
				const searchText = firstSectionText.slice(0, Math.min(80, firstSectionText.length));
				const foundIdx = cache.fullText.indexOf(searchText);
				if (foundIdx !== -1) {
					// Calculate offset between backend position and actual DOM position
					positionOffset = foundIdx - (firstSection.start_pos ?? 0);
				}
			}
		}

		// Phase 1: Find the START position of each section
		const sectionStarts: { section_num: number; startIdx: number; text: string }[] = [];
		let lastSearchPos = 0;

		for (const section of sortedSections) {
			const sectionText = normalizeHighlightText(section.text);
			if (!sectionText) continue;

			let startIdx = -1;

			// Try backend position first (with offset)
			if (backendPositionsValid && typeof section.start_pos === 'number') {
				const adjustedStart = section.start_pos + positionOffset;
				// Verify the position is correct by checking a snippet
				if (adjustedStart >= 0 && adjustedStart < cache.fullText.length) {
					const checkText = sectionText.slice(0, Math.min(40, sectionText.length));
					const actualText = cache.fullText.slice(adjustedStart, adjustedStart + checkText.length);
					// Case-insensitive comparison for flexibility
					if (
						actualText.toLowerCase() === checkText.toLowerCase() ||
						cache.fullText.slice(adjustedStart).toLowerCase().startsWith(checkText.toLowerCase())
					) {
						startIdx = adjustedStart;
					}
				}
			}

			// Fall back to text search if backend position didn't work
			if (startIdx === -1) {
				// Get the first paragraph/sentence to search for (more reliable than full text)
				const firstParagraph = sectionText.split(/\n+/).filter((p) => p.trim())[0] || sectionText;
				// Use first 100 chars of first paragraph for more reliable matching
				const searchText = firstParagraph.slice(0, Math.min(100, firstParagraph.length));

				// Search for start position
				startIdx = cache.fullText.indexOf(searchText, lastSearchPos);
				if (startIdx === -1) {
					// Try case-insensitive
					startIdx = cache.fullText.toLowerCase().indexOf(searchText.toLowerCase(), lastSearchPos);
				}
				if (startIdx === -1) {
					// Fallback: search from beginning
					startIdx = cache.fullText.indexOf(searchText);
					if (startIdx === -1) {
						startIdx = cache.fullText.toLowerCase().indexOf(searchText.toLowerCase());
					}
				}
			}

			if (startIdx !== -1) {
				sectionStarts.push({
					section_num: section.section_num,
					startIdx,
					text: sectionText
				});
				// Advance search position
				lastSearchPos = startIdx + Math.min(100, sectionText.length);
			}
		}

		// Phase 2: Determine END positions using the START of the next section
		for (let i = 0; i < sectionStarts.length; i++) {
			const current = sectionStarts[i];
			const next = sectionStarts[i + 1];

			let endIdx: number;
			if (next) {
				// End is where the next section starts
				endIdx = next.startIdx;
			} else {
				// Last section: find the actual end by searching for the last paragraph
				// Don't use fullText.length as it may include content after reasoning (e.g., final answer)
				const paragraphs = current.text.split(/\n+/).filter((p) => p.trim());
				const lastParagraph = paragraphs[paragraphs.length - 1] || current.text;
				// Use last 100 chars of last paragraph for reliable matching
				const searchText = lastParagraph.slice(-Math.min(100, lastParagraph.length));

				let lastIdx = cache.fullText.indexOf(searchText, current.startIdx);
				if (lastIdx === -1) {
					lastIdx = cache.fullText
						.toLowerCase()
						.indexOf(searchText.toLowerCase(), current.startIdx);
				}

				if (lastIdx !== -1) {
					endIdx = lastIdx + searchText.length;
				} else {
					// Fallback: estimate based on section text length from start
					endIdx = current.startIdx + current.text.length;
				}
			}

			positions[current.section_num] = {
				start: current.startIdx,
				end: endIdx
			};
		}

		return positions;
	};

	/**
	 * Ensure caches are built and up to date.
	 * Returns true if caches are valid.
	 */
	const ensureHighlightCaches = (
		sections?: { section_num: number; text: string; start_pos?: number; end_pos?: number }[]
	): boolean => {
		// Check if we need to rebuild text node cache
		// Ensure reasoning area is expanded first
		const isExpanded = ensureReasoningExpanded();
		if (!isExpanded) {
			// Content not yet rendered after expanding, return false to trigger retry
			return false;
		}

		const cacheKey = contentContainerElement ? `${message.id}:${message.content?.length ?? 0}` : '';

		if (!textNodeCache || textNodeCacheKey !== cacheKey) {
			textNodeCache = buildTextNodeCache();
			textNodeCacheKey = cacheKey;
			sectionPositionCache = {}; // Invalidate section cache
			sectionCacheKey = '';
		}

		// Check if we need to rebuild section position cache
		if (textNodeCache && sections && sections.length > 0) {
			const newSectionKey = `${cacheKey}:${sections.length}:${sections[0]?.text?.slice(0, 20) ?? ''}`;
			if (sectionCacheKey !== newSectionKey) {
				sectionPositionCache = buildSectionPositionCache(sections, textNodeCache);
				sectionCacheKey = newSectionKey;
			}
		}

		return textNodeCache !== null;
	};

	/**
	 * Fast highlight using pre-computed cache.
	 * Returns true if highlighting was successful.
	 */
	const highlightFromCache = (start: number, end: number, highlightClass: string): boolean => {
		if (!textNodeCache || start < 0 || end <= start) return false;

		const { textNodes } = textNodeCache;
		let hasHighlighted = false;
		let isFirstHighlight = true;

		for (const { node, length, startOffset } of textNodes) {
			if (!node) continue;

			const nodeEnd = startOffset + length;

			// Check if this node overlaps with highlight range
			if (nodeEnd <= start) continue; // Node is before highlight
			if (startOffset >= end) break; // Node is after highlight, done

			// Calculate overlap within this node
			const highlightStartInNode = Math.max(0, start - startOffset);
			const highlightEndInNode = Math.min(length, end - startOffset);

			if (highlightStartInNode < highlightEndInNode) {
				try {
					const range = document.createRange();
					range.setStart(node, highlightStartInNode);
					range.setEnd(node, highlightEndInNode);

					const mark = document.createElement('mark');
					mark.className = highlightClass;
					range.surroundContents(mark);
					reasoningHighlights.push(mark);

					if (isFirstHighlight) {
						mark.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
						isFirstHighlight = false;
					}

					hasHighlighted = true;
				} catch (e) {
					// Range error, skip
				}
			}
		}

		// Invalidate cache after modifying DOM
		if (hasHighlighted) {
			textNodeCache = null;
			textNodeCacheKey = '';
		}

		return hasHighlighted;
	};
	// ========== End Highlight Optimization ==========

	const clearReasoningHighlights = () => {
		if (!reasoningHighlights.length) return;
		reasoningHighlights.forEach((mark) => {
			const parent = mark.parentNode;
			if (parent) {
				parent.replaceChild(document.createTextNode(mark.textContent ?? ''), mark);
				parent.normalize();
			}
		});
		reasoningHighlights = [];
		// Invalidate cache since DOM changed
		textNodeCache = null;
		textNodeCacheKey = '';
	};

	const stripMarkdown = (text: string) => {
		if (!text) return '';
		// Remove bold/italic markers
		let cleaned = text.replace(/(\*\*|__)(.*?)\1/g, '$2');
		cleaned = cleaned.replace(/(\*|_)(.*?)\1/g, '$2');
		// Remove code ticks
		cleaned = cleaned.replace(/`([^`]+)`/g, '$1');
		// Remove links
		cleaned = cleaned.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
		return cleaned;
	};

	// Normalize text to match rendered content (strip markdown and blockquote markers, collapse whitespace)
	const normalizeHighlightText = (text: string) => {
		if (!text) return '';
		return stripMarkdown(text)
			.replace(/^\s*>[ \t]?/gm, '') // remove blockquote prefixes
			.replace(/[ \t]+/g, ' ') // collapse inline spaces (not newlines)
			.replace(/\n+/g, '\n') // collapse multiple newlines to single
			.replace(/\s+\n/g, '\n')
			.replace(/\n\s+/g, '\n')
			.trim();
	};

	// Pending highlight request for retry logic
	let pendingHighlightRequest: {
		sectionStart?: number | null;
		sectionEnd?: number | null;
		sections?: { section_num: number; text: string; start_pos?: number; end_pos?: number }[];
		isError?: boolean;
		sectionNumbers?: number[];
		highlightType?: string;
		retryCount: number;
	} | null = null;
	let highlightRetryTimer: ReturnType<typeof setTimeout> | null = null;

	/**
	 * Highlight sections in the reasoning content.
	 * Uses a pre-built cache for fast highlighting with retry logic if DOM isn't ready.
	 *
	 * @param _sentence - Deprecated, kept for backward compatibility (unused)
	 * @param sectionStart - Start section number (inclusive)
	 * @param sectionEnd - End section number (inclusive)
	 * @param sections - Array of sections from analysis result
	 * @param isError - If true, use error highlight style (red underline)
	 * @param sectionNumbers - Specific section numbers to highlight (alternative to range)
	 * @param highlightType - 'error' for red, 'answer' for purple (future use)
	 */
	const highlightReasoningSentence = (
		_sentence: string,
		sectionStart?: number | null,
		sectionEnd?: number | null,
		sections?: { section_num: number; text: string; start_pos?: number; end_pos?: number }[],
		isError?: boolean,
		sectionNumbers?: number[],
		highlightType?: string
	) => {
		// Clear any pending retry
		if (highlightRetryTimer) {
			clearTimeout(highlightRetryTimer);
			highlightRetryTimer = null;
		}
		pendingHighlightRequest = null;

		// Determine highlight class based on type
		let highlightClass = 'reasoning-highlight';
		if (isError) {
			highlightClass = 'reasoning-error-highlight';
		} else if (highlightType === 'answer') {
			highlightClass = 'reasoning-answer-highlight';
		}

		const hasSectionNumbers =
			sectionNumbers && sectionNumbers.length > 0 && sections && sections.length > 0;
		const hasSections =
			sectionStart != null && sectionEnd != null && sections && sections.length > 0;

		clearReasoningHighlights();

		if (!contentContainerElement) {
			// Schedule retry if container not ready yet
			if (hasSectionNumbers || hasSections) {
				scheduleHighlightRetry({
					sectionStart,
					sectionEnd,
					sections,
					isError,
					sectionNumbers,
					highlightType,
					retryCount: 0
				});
			}
			return;
		}

		// Only proceed if we have section-based data
		if (!hasSectionNumbers && !hasSections) {
			return;
		}

		// Build or reuse cached section positions for fast highlighting
		if (sections) {
			ensureHighlightCaches(sections);
		}

		// Check if cache building failed (DOM might not be ready)
		if (!textNodeCache || textNodeCache.fullText.length === 0) {
			// Schedule retry - DOM might not be fully rendered yet
			scheduleHighlightRetry({
				sectionStart,
				sectionEnd,
				sections,
				isError,
				sectionNumbers,
				highlightType,
				retryCount: 0
			});
			return;
		}

		// Use cached positions for fast highlighting
		if (textNodeCache && Object.keys(sectionPositionCache).length > 0) {
			let highlightRanges: { start: number; end: number }[] = [];

			if (hasSectionNumbers) {
				// Collect all section ranges from sectionNumbers
				for (const sectionNum of sectionNumbers!) {
					const pos = sectionPositionCache[sectionNum];
					if (pos) {
						highlightRanges.push(pos);
					}
				}
			} else if (hasSections) {
				// Collect range from sectionStart to sectionEnd
				for (let num = sectionStart!; num <= sectionEnd!; num++) {
					const pos = sectionPositionCache[num];
					if (pos) {
						highlightRanges.push(pos);
					}
				}
			}

			// If no positions found in cache, maybe cache is stale - schedule retry
			if (highlightRanges.length === 0 && (hasSectionNumbers || hasSections)) {
				// Invalidate cache and retry
				textNodeCache = null;
				textNodeCacheKey = '';
				sectionPositionCache = {};
				sectionCacheKey = '';
				scheduleHighlightRetry({
					sectionStart,
					sectionEnd,
					sections,
					isError,
					sectionNumbers,
					highlightType,
					retryCount: 0
				});
				return;
			}

			// Merge consecutive ranges and highlight
			if (highlightRanges.length > 0) {
				// Sort by start position
				highlightRanges.sort((a, b) => a.start - b.start);

				// Merge overlapping/adjacent ranges
				const mergedRanges: { start: number; end: number }[] = [];
				let current = highlightRanges[0];

				for (let i = 1; i < highlightRanges.length; i++) {
					const next = highlightRanges[i];
					// Allow small gap between sections (up to 50 chars for paragraph breaks)
					if (next.start <= current.end + 50) {
						current = { start: current.start, end: Math.max(current.end, next.end) };
					} else {
						mergedRanges.push(current);
						current = next;
					}
				}
				mergedRanges.push(current);

				// Highlight each merged range
				for (const range of mergedRanges) {
					highlightFromCache(range.start, range.end, highlightClass);
				}
			}
		} else if (hasSectionNumbers || hasSections) {
			// Cache is empty but we have section data - schedule retry
			scheduleHighlightRetry({
				sectionStart,
				sectionEnd,
				sections,
				isError,
				sectionNumbers,
				highlightType,
				retryCount: 0
			});
		}
	};

	const scheduleHighlightRetry = (request: NonNullable<typeof pendingHighlightRequest>) => {
		const MAX_RETRIES = 3;
		const RETRY_DELAY = 150; // ms

		if (request.retryCount >= MAX_RETRIES) {
			// Give up after max retries
			return;
		}

		pendingHighlightRequest = { ...request, retryCount: request.retryCount + 1 };
		highlightRetryTimer = setTimeout(() => {
			if (pendingHighlightRequest) {
				const req = pendingHighlightRequest;
				pendingHighlightRequest = null;
				highlightRetryTimer = null;
				// Retry with incremented count
				highlightReasoningSentence(
					'',
					req.sectionStart,
					req.sectionEnd,
					req.sections,
					req.isError,
					req.sectionNumbers,
					req.highlightType
				);
			}
		}, RETRY_DELAY);
	};

	onMount(async () => {
		await tick();
		if (typeof window !== 'undefined') {
			window.addEventListener('resize', handleWindowResize);
			window.addEventListener('analysis-panel-open', analysisPanelOpenHandler as EventListener);
		}

		if (buttonsContainerElement) {
			buttonsContainerElement.addEventListener('wheel', buttonsWheelHandler);
		}

		if (contentContainerElement) {
			contentContainerElement.addEventListener('copy', contentCopyHandler);
		}
	});

	onDestroy(() => {
		if (typeof window !== 'undefined') {
			window.removeEventListener('resize', handleWindowResize);
			window.removeEventListener('analysis-panel-open', analysisPanelOpenHandler as EventListener);
		}

		if (buttonsContainerElement) {
			buttonsContainerElement.removeEventListener('wheel', buttonsWheelHandler);
		}

		if (contentContainerElement) {
			contentContainerElement.removeEventListener('copy', contentCopyHandler);
		}

		// reset inset when component unmounts
		if (typeof document !== 'undefined') {
			document.documentElement.style.setProperty(analysisInsetVarName, '0px');
		}

		// Note: We intentionally do NOT cancel the analysis task here.
		// Analysis runs in the global store and continues even when user navigates away.
		// This enables background analysis functionality.
		stopPanelResize();
		clearReasoningHighlights();

		// Clean up highlight retry timer
		if (highlightRetryTimer) {
			clearTimeout(highlightRetryTimer);
			highlightRetryTimer = null;
		}
		pendingHighlightRequest = null;
	});
</script>

<DeleteConfirmDialog
	bind:show={showDeleteConfirm}
	title={$i18n.t('Delete message?')}
	on:confirm={() => {
		deleteMessageHandler();
	}}
/>

{#key message.id}
	<div
		class=" flex w-full message-{message.id}"
		id="message-{message.id}"
		dir={$settings.chatDirection}
	>
		<div class={`shrink-0 ltr:mr-3 rtl:ml-3 hidden @lg:flex mt-1 `}>
			<ProfileImage
				src={`${WEBUI_API_BASE_URL}/models/model/profile/image?id=${model?.id}&lang=${$i18n.language}`}
				className={'size-8 assistant-message-profile-image'}
			/>
		</div>

		<div class="flex-auto min-w-0 pl-1 relative">
			<Name>
				<Tooltip content={model?.name ?? message.model} placement="top-start">
					<span id="response-message-model-name" class="line-clamp-1 text-black dark:text-white">
						{model?.name ?? message.model}
					</span>
				</Tooltip>

				{#if message.timestamp}
					<div
						class="self-center text-xs font-medium first-letter:capitalize ml-0.5 translate-y-[1px] {($settings?.highContrastMode ??
						false)
							? 'dark:text-gray-100 text-gray-900'
							: 'invisible group-hover:visible transition text-gray-400'}"
					>
						<Tooltip content={dayjs(message.timestamp * 1000).format('LLLL')}>
							<span class="line-clamp-1"
								>{$i18n.t(formatDate(message.timestamp * 1000), {
									LOCALIZED_TIME: dayjs(message.timestamp * 1000).format('LT'),
									LOCALIZED_DATE: dayjs(message.timestamp * 1000).format('L')
								})}</span
							>
						</Tooltip>
					</div>
				{/if}
			</Name>

			<div>
				<div class="chat-{message.role} w-full min-w-full markdown-prose">
					<div>
						{#if model?.info?.meta?.capabilities?.status_updates ?? true}
							<StatusHistory statusHistory={message?.statusHistory} />
						{/if}

						{#if message?.files && message.files?.filter((f) => f.type === 'image').length > 0}
							<div class="my-1 w-full flex overflow-x-auto gap-2 flex-wrap">
								{#each message.files as file}
									<div>
										{#if file.type === 'image'}
											<Image src={file.url} alt={message.content} />
										{:else}
											<FileItem
												item={file}
												url={file.url}
												name={file.name}
												type={file.type}
												size={file?.size}
												small={true}
											/>
										{/if}
									</div>
								{/each}
							</div>
						{/if}

						{#if message?.embeds && message.embeds.length > 0}
							<div class="my-1 w-full flex overflow-x-auto gap-2 flex-wrap">
								{#each message.embeds as embed, idx}
									<div class="my-2 w-full" id={`${message.id}-embeds-${idx}`}>
										<FullHeightIframe
											src={embed}
											allowScripts={true}
											allowForms={true}
											allowSameOrigin={true}
											allowPopups={true}
										/>
									</div>
								{/each}
							</div>
						{/if}

						{#if edit === true}
							<div class="w-full bg-gray-50 dark:bg-gray-800 rounded-3xl px-5 py-3 my-2">
								<textarea
									id="message-edit-{message.id}"
									bind:this={editTextAreaElement}
									class=" bg-transparent outline-hidden w-full resize-none"
									bind:value={editedContent}
									on:input={(e) => {
										e.target.style.height = '';
										e.target.style.height = `${e.target.scrollHeight}px`;
									}}
									on:keydown={(e) => {
										if (e.key === 'Escape') {
											document.getElementById('close-edit-message-button')?.click();
										}

										const isCmdOrCtrlPressed = e.metaKey || e.ctrlKey;
										const isEnterPressed = e.key === 'Enter';

										if (isCmdOrCtrlPressed && isEnterPressed) {
											document.getElementById('confirm-edit-message-button')?.click();
										}
									}}
								/>

								<div class=" mt-2 mb-1 flex justify-between text-sm font-medium">
									<div>
										<button
											id="save-new-message-button"
											class="px-3.5 py-1.5 bg-gray-50 hover:bg-gray-100 dark:bg-gray-800 dark:hover:bg-gray-700 border border-gray-100 dark:border-gray-700 text-gray-700 dark:text-gray-200 transition rounded-3xl"
											on:click={() => {
												saveAsCopyHandler();
											}}
										>
											{$i18n.t('Save As Copy')}
										</button>
									</div>

									<div class="flex space-x-1.5">
										<button
											id="close-edit-message-button"
											class="px-3.5 py-1.5 bg-white dark:bg-gray-900 hover:bg-gray-100 text-gray-800 dark:text-gray-100 transition rounded-3xl"
											on:click={() => {
												cancelEditMessage();
											}}
										>
											{$i18n.t('Cancel')}
										</button>

										<button
											id="confirm-edit-message-button"
											class="px-3.5 py-1.5 bg-gray-900 dark:bg-white hover:bg-gray-850 text-gray-100 dark:text-gray-800 transition rounded-3xl"
											on:click={() => {
												editMessageConfirmHandler();
											}}
										>
											{$i18n.t('Save')}
										</button>
									</div>
								</div>
							</div>
						{/if}

						<div
							bind:this={contentContainerElement}
							class="w-full flex flex-col relative {edit ? 'hidden' : ''}"
							id="response-content-container"
						>
							{#if message.content === '' && !message.error && ((model?.info?.meta?.capabilities?.status_updates ?? true) ? (message?.statusHistory ?? [...(message?.status ? [message?.status] : [])]).length === 0 || (message?.statusHistory?.at(-1)?.hidden ?? false) : true)}
								<Skeleton />
							{:else if message.content && message.error !== true}
								<!-- always show message contents even if there's an error -->
								<!-- unless message.error === true which is legacy error handling, where the error message is stored in message.content -->
								<ContentRenderer
									id={`${chatId}-${message.id}`}
									messageId={message.id}
									{history}
									{selectedModels}
									content={message.content}
									sources={message.sources}
									floatingButtons={message?.done &&
										!readOnly &&
										($settings?.showFloatingActionButtons ?? true)}
									save={!readOnly}
									preview={!readOnly}
									{editCodeBlock}
									{topPadding}
									done={($settings?.chatFadeStreamingText ?? true)
										? (message?.done ?? false)
										: true}
									{model}
									onTaskClick={async (e) => {
										console.log(e);
									}}
									onSourceClick={async (id) => {
										console.log(id);

										if (citationsElement) {
											citationsElement?.showSourceModal(id);
										}
									}}
									onAddMessages={({ modelId, parentId, messages }) => {
										addMessages({ modelId, parentId, messages });
									}}
									onSave={({ raw, oldContent, newContent }) => {
										history.messages[message.id].content = history.messages[
											message.id
										].content.replace(raw, raw.replace(oldContent, newContent));

										updateChat();
									}}
									on:analyzeReasoning={() => {
										handleAnalyzeReasoning();
									}}
								/>
							{/if}

							{#if message?.error}
								<ErrorMessage content={message?.error?.content ?? message.content} />
							{/if}

							{#if (message?.sources || message?.citations) && (model?.info?.meta?.capabilities?.citations ?? true)}
								<Citations
									bind:this={citationsElement}
									id={message?.id}
									sources={message?.sources ?? message?.citations}
									{readOnly}
								/>
							{/if}

							{#if message.code_executions}
								<CodeExecutions codeExecutions={message.code_executions} />
							{/if}
						</div>
					</div>
				</div>

				{#if !edit}
					<div
						bind:this={buttonsContainerElement}
						class="flex justify-start overflow-x-auto buttons text-gray-600 dark:text-gray-500 mt-0.5"
					>
						{#if message.done || siblings.length > 1}
							{#if siblings.length > 1}
								<div class="flex self-center min-w-fit" dir="ltr">
									<button
										aria-label={$i18n.t('Previous message')}
										class="self-center p-1 hover:bg-black/5 dark:hover:bg-white/5 dark:hover:text-white hover:text-black rounded-md transition"
										on:click={() => {
											showPreviousMessage(message);
										}}
									>
										<svg
											aria-hidden="true"
											xmlns="http://www.w3.org/2000/svg"
											fill="none"
											viewBox="0 0 24 24"
											stroke="currentColor"
											stroke-width="2.5"
											class="size-3.5"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												d="M15.75 19.5 8.25 12l7.5-7.5"
											/>
										</svg>
									</button>

									{#if messageIndexEdit}
										<div
											class="text-sm flex justify-center font-semibold self-center dark:text-gray-100 min-w-fit"
										>
											<input
												id="message-index-input-{message.id}"
												type="number"
												value={siblings.indexOf(message.id) + 1}
												min="1"
												max={siblings.length}
												on:focus={(e) => {
													e.target.select();
												}}
												on:blur={(e) => {
													gotoMessage(message, e.target.value - 1);
													messageIndexEdit = false;
												}}
												on:keydown={(e) => {
													if (e.key === 'Enter') {
														gotoMessage(message, e.target.value - 1);
														messageIndexEdit = false;
													}
												}}
												class="bg-transparent font-semibold self-center dark:text-gray-100 min-w-fit outline-hidden"
											/>/{siblings.length}
										</div>
									{:else}
										<!-- svelte-ignore a11y-no-static-element-interactions -->
										<div
											class="text-sm tracking-widest font-semibold self-center dark:text-gray-100 min-w-fit"
											on:dblclick={async () => {
												messageIndexEdit = true;

												await tick();
												const input = document.getElementById(`message-index-input-${message.id}`);
												if (input) {
													input.focus();
													input.select();
												}
											}}
										>
											{siblings.indexOf(message.id) + 1}/{siblings.length}
										</div>
									{/if}

									<button
										class="self-center p-1 hover:bg-black/5 dark:hover:bg-white/5 dark:hover:text-white hover:text-black rounded-md transition"
										on:click={() => {
											showNextMessage(message);
										}}
										aria-label={$i18n.t('Next message')}
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											fill="none"
											aria-hidden="true"
											viewBox="0 0 24 24"
											stroke="currentColor"
											stroke-width="2.5"
											class="size-3.5"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												d="m8.25 4.5 7.5 7.5-7.5 7.5"
											/>
										</svg>
									</button>
								</div>
							{/if}

							{#if message.done}
								{#if !readOnly}
									{#if $user?.role === 'user' ? ($user?.permissions?.chat?.edit ?? true) : true}
										<Tooltip content={$i18n.t('Edit')} placement="bottom">
											<button
												aria-label={$i18n.t('Edit')}
												class="{isLastMessage || ($settings?.highContrastMode ?? false)
													? 'visible'
													: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition"
												on:click={() => {
													editMessageHandler();
												}}
											>
												<svg
													xmlns="http://www.w3.org/2000/svg"
													fill="none"
													viewBox="0 0 24 24"
													stroke-width="2.3"
													aria-hidden="true"
													stroke="currentColor"
													class="w-4 h-4"
												>
													<path
														stroke-linecap="round"
														stroke-linejoin="round"
														d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487zm0 0L19.5 7.125"
													/>
												</svg>
											</button>
										</Tooltip>
									{/if}
								{/if}

								<Tooltip content={$i18n.t('Copy')} placement="bottom">
									<button
										aria-label={$i18n.t('Copy')}
										class="{isLastMessage || ($settings?.highContrastMode ?? false)
											? 'visible'
											: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition copy-response-button"
										on:click={() => {
											copyToClipboard(message.content);
										}}
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											fill="none"
											aria-hidden="true"
											viewBox="0 0 24 24"
											stroke-width="2.3"
											stroke="currentColor"
											class="w-4 h-4"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184"
											/>
										</svg>
									</button>
								</Tooltip>

								<Tooltip content={$i18n.t('Analyze')} placement="bottom">
									<button
										aria-label={$i18n.t('Analyze')}
										class="{isLastMessage || ($settings?.highContrastMode ?? false)
											? 'visible'
											: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition"
										disabled={analysisLoading && showAnalysisPanel}
										on:click={() => handleAnalyzeReasoning()}
									>
										{#if analysisLoading}
											<Spinner className="size-4" />
										{:else}
											<Sparkles strokeWidth="2.1" className="size-4" />
										{/if}
									</button>
								</Tooltip>

								{#if $user?.role === 'admin' || ($user?.permissions?.chat?.tts ?? true)}
									<Tooltip content={$i18n.t('Read Aloud')} placement="bottom">
										<button
											aria-label={$i18n.t('Read Aloud')}
											id="speak-button-{message.id}"
											class="{isLastMessage || ($settings?.highContrastMode ?? false)
												? 'visible'
												: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition"
											on:click={() => {
												if (!loadingSpeech) {
													if (speaking) {
														stopAudio();
													} else {
														speak();
													}
												}
											}}
										>
											{#if loadingSpeech}
												<svg
													class=" w-4 h-4"
													fill="currentColor"
													viewBox="0 0 24 24"
													aria-hidden="true"
													xmlns="http://www.w3.org/2000/svg"
												>
													<style>
														.spinner_S1WN {
															animation: spinner_MGfb 0.8s linear infinite;
															animation-delay: -0.8s;
														}

														.spinner_Km9P {
															animation-delay: -0.65s;
														}

														.spinner_JApP {
															animation-delay: -0.5s;
														}

														@keyframes spinner_MGfb {
															93.75%,
															100% {
																opacity: 0.2;
															}
														}
													</style>
													<circle class="spinner_S1WN" cx="4" cy="12" r="3" />
													<circle class="spinner_S1WN spinner_Km9P" cx="12" cy="12" r="3" />
													<circle class="spinner_S1WN spinner_JApP" cx="20" cy="12" r="3" />
												</svg>
											{:else if speaking}
												<svg
													xmlns="http://www.w3.org/2000/svg"
													fill="none"
													viewBox="0 0 24 24"
													aria-hidden="true"
													stroke-width="2.3"
													stroke="currentColor"
													class="w-4 h-4"
												>
													<path
														stroke-linecap="round"
														stroke-linejoin="round"
														d="M17.25 9.75 19.5 12m0 0 2.25 2.25M19.5 12l2.25-2.25M19.5 12l-2.25 2.25m-10.5-6 4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z"
													/>
												</svg>
											{:else}
												<svg
													xmlns="http://www.w3.org/2000/svg"
													fill="none"
													viewBox="0 0 24 24"
													aria-hidden="true"
													stroke-width="2.3"
													stroke="currentColor"
													class="w-4 h-4"
												>
													<path
														stroke-linecap="round"
														stroke-linejoin="round"
														d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z"
													/>
												</svg>
											{/if}
										</button>
									</Tooltip>
								{/if}

								{#if $config?.features.enable_image_generation && ($user?.role === 'admin' || $user?.permissions?.features?.image_generation) && !readOnly}
									<Tooltip content={$i18n.t('Generate Image')} placement="bottom">
										<button
											aria-label={$i18n.t('Generate Image')}
											class="{isLastMessage || ($settings?.highContrastMode ?? false)
												? 'visible'
												: 'invisible group-hover:visible'}  p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition"
											on:click={() => {
												if (!generatingImage) {
													generateImage(message);
												}
											}}
										>
											{#if generatingImage}
												<svg
													aria-hidden="true"
													class=" w-4 h-4"
													fill="currentColor"
													viewBox="0 0 24 24"
													xmlns="http://www.w3.org/2000/svg"
												>
													<style>
														.spinner_S1WN {
															animation: spinner_MGfb 0.8s linear infinite;
															animation-delay: -0.8s;
														}

														.spinner_Km9P {
															animation-delay: -0.65s;
														}

														.spinner_JApP {
															animation-delay: -0.5s;
														}

														@keyframes spinner_MGfb {
															93.75%,
															100% {
																opacity: 0.2;
															}
														}
													</style>
													<circle class="spinner_S1WN" cx="4" cy="12" r="3" />
													<circle class="spinner_S1WN spinner_Km9P" cx="12" cy="12" r="3" />
													<circle class="spinner_S1WN spinner_JApP" cx="20" cy="12" r="3" />
												</svg>
											{:else}
												<svg
													xmlns="http://www.w3.org/2000/svg"
													fill="none"
													aria-hidden="true"
													viewBox="0 0 24 24"
													stroke-width="2.3"
													stroke="currentColor"
													class="w-4 h-4"
												>
													<path
														stroke-linecap="round"
														stroke-linejoin="round"
														d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Zm10.5-11.25h.008v.008h-.008V8.25Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z"
													/>
												</svg>
											{/if}
										</button>
									</Tooltip>
								{/if}

								{#if message.usage}
									<Tooltip
										content={message.usage
											? `<pre>${sanitizeResponseContent(
													JSON.stringify(message.usage, null, 2)
														.replace(/"([^(")"]+)":/g, '$1:')
														.slice(1, -1)
														.split('\n')
														.map((line) => line.slice(2))
														.map((line) => (line.endsWith(',') ? line.slice(0, -1) : line))
														.join('\n')
												)}</pre>`
											: ''}
										placement="bottom"
									>
										<button
											aria-hidden="true"
											class=" {isLastMessage || ($settings?.highContrastMode ?? false)
												? 'visible'
												: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition whitespace-pre-wrap"
											on:click={() => {
												console.log(message);
											}}
											id="info-{message.id}"
										>
											<svg
												aria-hidden="true"
												xmlns="http://www.w3.org/2000/svg"
												fill="none"
												viewBox="0 0 24 24"
												stroke-width="2.3"
												stroke="currentColor"
												class="w-4 h-4"
											>
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z"
												/>
											</svg>
										</button>
									</Tooltip>
								{/if}

								{#if !readOnly}
									{#if !$temporaryChatEnabled && ($config?.features.enable_message_rating ?? true) && ($user?.role === 'admin' || ($user?.permissions?.chat?.rate_response ?? true))}
										<Tooltip content={$i18n.t('Good Response')} placement="bottom">
											<button
												aria-label={$i18n.t('Good Response')}
												class="{isLastMessage || ($settings?.highContrastMode ?? false)
													? 'visible'
													: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg {(
													message?.annotation?.rating ?? ''
												).toString() === '1'
													? 'bg-gray-100 dark:bg-gray-800'
													: ''} dark:hover:text-white hover:text-black transition disabled:cursor-progress disabled:hover:bg-transparent"
												disabled={feedbackLoading}
												on:click={async () => {
													await feedbackHandler(1);
													window.setTimeout(() => {
														document
															.getElementById(`message-feedback-${message.id}`)
															?.scrollIntoView();
													}, 0);
												}}
											>
												<svg
													aria-hidden="true"
													stroke="currentColor"
													fill="none"
													stroke-width="2.3"
													viewBox="0 0 24 24"
													stroke-linecap="round"
													stroke-linejoin="round"
													class="w-4 h-4"
													xmlns="http://www.w3.org/2000/svg"
												>
													<path
														d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"
													/>
												</svg>
											</button>
										</Tooltip>

										<Tooltip content={$i18n.t('Bad Response')} placement="bottom">
											<button
												aria-label={$i18n.t('Bad Response')}
												class="{isLastMessage || ($settings?.highContrastMode ?? false)
													? 'visible'
													: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg {(
													message?.annotation?.rating ?? ''
												).toString() === '-1'
													? 'bg-gray-100 dark:bg-gray-800'
													: ''} dark:hover:text-white hover:text-black transition disabled:cursor-progress disabled:hover:bg-transparent"
												disabled={feedbackLoading}
												on:click={async () => {
													await feedbackHandler(-1);
													window.setTimeout(() => {
														document
															.getElementById(`message-feedback-${message.id}`)
															?.scrollIntoView();
													}, 0);
												}}
											>
												<svg
													aria-hidden="true"
													stroke="currentColor"
													fill="none"
													stroke-width="2.3"
													viewBox="0 0 24 24"
													stroke-linecap="round"
													stroke-linejoin="round"
													class="w-4 h-4"
													xmlns="http://www.w3.org/2000/svg"
												>
													<path
														d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"
													/>
												</svg>
											</button>
										</Tooltip>
									{/if}

									{#if isLastMessage && ($user?.role === 'admin' || ($user?.permissions?.chat?.continue_response ?? true))}
										<Tooltip content={$i18n.t('Continue Response')} placement="bottom">
											<button
												aria-label={$i18n.t('Continue Response')}
												type="button"
												id="continue-response-button"
												class="{isLastMessage || ($settings?.highContrastMode ?? false)
													? 'visible'
													: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition"
												on:click={() => {
													continueResponse();
												}}
											>
												<svg
													aria-hidden="true"
													xmlns="http://www.w3.org/2000/svg"
													fill="none"
													viewBox="0 0 24 24"
													stroke-width="2.3"
													stroke="currentColor"
													class="w-4 h-4"
												>
													<path
														stroke-linecap="round"
														stroke-linejoin="round"
														d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
													/>
													<path
														stroke-linecap="round"
														stroke-linejoin="round"
														d="M15.91 11.672a.375.375 0 0 1 0 .656l-5.603 3.113a.375.375 0 0 1-.557-.328V8.887c0-.286.307-.466.557-.327l5.603 3.112Z"
													/>
												</svg>
											</button>
										</Tooltip>
									{/if}

									{#if $user?.role === 'admin' || ($user?.permissions?.chat?.regenerate_response ?? true)}
										{#if $settings?.regenerateMenu ?? true}
											<button
												type="button"
												class="hidden regenerate-response-button"
												on:click={() => {
													showRateComment = false;
													regenerateResponse(message);

													(model?.actions ?? []).forEach((action) => {
														dispatch('action', {
															id: action.id,
															event: {
																id: 'regenerate-response',
																data: {
																	messageId: message.id
																}
															}
														});
													});
												}}
											/>

											<RegenerateMenu
												onRegenerate={(prompt = null) => {
													showRateComment = false;
													regenerateResponse(message, prompt);

													(model?.actions ?? []).forEach((action) => {
														dispatch('action', {
															id: action.id,
															event: {
																id: 'regenerate-response',
																data: {
																	messageId: message.id
																}
															}
														});
													});
												}}
											>
												<Tooltip content={$i18n.t('Regenerate')} placement="bottom">
													<div
														aria-label={$i18n.t('Regenerate')}
														class="{isLastMessage
															? 'visible'
															: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition"
													>
														<svg
															xmlns="http://www.w3.org/2000/svg"
															fill="none"
															viewBox="0 0 24 24"
															stroke-width="2.3"
															aria-hidden="true"
															stroke="currentColor"
															class="w-4 h-4"
														>
															<path
																stroke-linecap="round"
																stroke-linejoin="round"
																d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
															/>
														</svg>
													</div>
												</Tooltip>
											</RegenerateMenu>
										{:else}
											<Tooltip content={$i18n.t('Regenerate')} placement="bottom">
												<button
													type="button"
													aria-label={$i18n.t('Regenerate')}
													class="{isLastMessage
														? 'visible'
														: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition regenerate-response-button"
													on:click={() => {
														showRateComment = false;
														regenerateResponse(message);

														(model?.actions ?? []).forEach((action) => {
															dispatch('action', {
																id: action.id,
																event: {
																	id: 'regenerate-response',
																	data: {
																		messageId: message.id
																	}
																}
															});
														});
													}}
												>
													<svg
														xmlns="http://www.w3.org/2000/svg"
														fill="none"
														viewBox="0 0 24 24"
														stroke-width="2.3"
														aria-hidden="true"
														stroke="currentColor"
														class="w-4 h-4"
													>
														<path
															stroke-linecap="round"
															stroke-linejoin="round"
															d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
														/>
													</svg>
												</button>
											</Tooltip>
										{/if}
									{/if}

									{#if $user?.role === 'admin' || ($user?.permissions?.chat?.delete_message ?? true)}
										{#if siblings.length > 1}
											<Tooltip content={$i18n.t('Delete')} placement="bottom">
												<button
													type="button"
													aria-label={$i18n.t('Delete')}
													id="delete-response-button"
													class="{isLastMessage || ($settings?.highContrastMode ?? false)
														? 'visible'
														: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition"
													on:click={() => {
														showDeleteConfirm = true;
													}}
												>
													<svg
														xmlns="http://www.w3.org/2000/svg"
														fill="none"
														viewBox="0 0 24 24"
														stroke-width="2"
														stroke="currentColor"
														aria-hidden="true"
														class="w-4 h-4"
													>
														<path
															stroke-linecap="round"
															stroke-linejoin="round"
															d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
														/>
													</svg>
												</button>
											</Tooltip>
										{/if}
									{/if}

									{#if isLastMessage}
										{#each model?.actions ?? [] as action}
											<Tooltip content={action.name} placement="bottom">
												<button
													type="button"
													aria-label={action.name}
													class="{isLastMessage || ($settings?.highContrastMode ?? false)
														? 'visible'
														: 'invisible group-hover:visible'} p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-lg dark:hover:text-white hover:text-black transition"
													on:click={() => {
														actionMessage(action.id, message);
													}}
												>
													{#if action?.icon}
														<div class="size-4">
															<img
																src={action.icon}
																class="w-4 h-4 {action.icon.includes('svg')
																	? 'dark:invert-[80%]'
																	: ''}"
																style="fill: currentColor;"
																alt={action.name}
															/>
														</div>
													{:else}
														<Sparkles strokeWidth="2.1" className="size-4" />
													{/if}
												</button>
											</Tooltip>
										{/each}
									{/if}
								{/if}
							{/if}
						{/if}
					</div>

					{#if message.done && showRateComment}
						<RateComment
							bind:message
							bind:show={showRateComment}
							on:save={async (e) => {
								await feedbackHandler(null, {
									...e.detail
								});
							}}
						/>
					{/if}

					{#if (isLastMessage || ($settings?.keepFollowUpPrompts ?? false)) && message.done && !readOnly && (message?.followUps ?? []).length > 0}
						<div class="mt-2.5" in:fade={{ duration: 100 }}>
							<FollowUps
								followUps={message?.followUps}
								onClick={(prompt) => {
									if ($settings?.insertFollowUpPrompt ?? false) {
										// Insert the follow-up prompt into the input box
										setInputText(prompt);
									} else {
										// Submit the follow-up prompt directly
										submitMessage(message?.id, prompt);
									}
								}}
							/>
						</div>
					{/if}
				{/if}
			</div>
		</div>
	</div>

	{#if showAnalysisPanel}
		<div
			class={`fixed inset-0 z-50 ${$mobile ? 'bg-black/40' : 'pointer-events-none'}`}
			on:click={() => {
				if ($mobile) {
					closeAnalysisPanel();
				}
			}}
		>
			<div
				class={`relative h-full w-full flex justify-end ${$mobile ? '' : 'pointer-events-none'}`}
				on:click|stopPropagation={() => {}}
			>
				<div
					class="relative h-full w-full sm:w-[460px] md:w-[560px] pointer-events-auto bg-white dark:bg-gray-900 dark:text-gray-50 shadow-2xl flex flex-col border border-gray-200 dark:border-gray-800 md:rounded-l-2xl overflow-hidden ml-auto"
					style={!$mobile
						? `width: ${analysisCollapsed ? analysisCollapsedPeekPx : analysisPanelWidth}px; transition: width 220ms ease;`
						: ''}
				>
					{#if !$mobile && !analysisCollapsed}
						<div
							class="absolute left-0 top-0 bottom-0 w-1.5 cursor-ew-resize group/resizer"
							on:pointerdown={startPanelResize}
							title={$i18n.t('Drag to resize')}
						>
							<div
								class="absolute left-0 top-1/4 bottom-1/4 w-1 bg-gray-300/60 dark:bg-gray-700/60 group-hover/resizer:bg-gray-500"
							></div>
						</div>
					{/if}

					{#if analysisCollapsed && !$mobile}
						<div
							class="flex h-full w-full flex-col items-center justify-center gap-3 p-3 text-[11px] text-gray-600 dark:text-gray-300"
						>
							<div class="font-semibold text-gray-700 dark:text-gray-100">
								{$i18n.t('Analysis')}
							</div>
							<div class="flex gap-2">
								<button
									class="p-2 rounded-full border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
									on:click={toggleAnalysisCollapse}
									aria-label={$i18n.t('Expand panel')}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="2"
										stroke="currentColor"
										class="size-4"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="m14.25 6.75-4.5 4.5 4.5 4.5"
										/>
									</svg>
								</button>
								<button
									class="p-2 rounded-full border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
									on:click={closeAnalysisPanel}
									aria-label={$i18n.t('Close')}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="2"
										stroke="currentColor"
										class="size-4"
									>
										<path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
									</svg>
								</button>
							</div>
						</div>
					{:else}
						<div class="px-4 py-3 border-b border-gray-200 dark:border-gray-800 space-y-2">
							<div class="flex items-start justify-between gap-3">
								<div class="space-y-1">
									<div class="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
										{$i18n.t('Reasoning Analysis')}
									</div>
									<div class="min-w-[200px]" data-analysis-no-drag>
										<Selector
											id="analysis-model-selector"
											className="w-full"
											triggerClassName="text-sm font-semibold"
											placeholder={$i18n.t('Select a model')}
											items={$models.map((m) => ({
												value: m.id,
												label: m.name,
												model: m
											}))}
											bind:value={analysisModelId}
										/>
									</div>
								</div>

								<div class="flex items-center gap-2">
									<button
										class="px-3 py-1.5 text-xs rounded-full border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-60 flex items-center gap-2 min-w-[180px] justify-center whitespace-nowrap"
										disabled={analysisLoading && showAnalysisPanel}
										on:click={() => {
											if (analysisLoading) {
												// If analysis is in progress, open panel to show progress
												openAnalysisPanel();
											} else {
												// Otherwise start new analysis
												startAnalysis();
											}
										}}
									>
										{#if analysisLoading}
											<Spinner className="size-3.5" />
											<span class="truncate">{analysisPrimaryStatus}</span>
										{:else}
											{$i18n.t('Analyze')}
										{/if}
									</button>
									{#if !$mobile}
										<button
											class="p-2 rounded-full border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
											on:click={toggleAnalysisCollapse}
											aria-label={analysisCollapsed
												? $i18n.t('Expand panel')
												: $i18n.t('Collapse panel')}
											title={analysisCollapsed
												? $i18n.t('Expand panel')
												: $i18n.t('Collapse panel')}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												fill="none"
												viewBox="0 0 24 24"
												stroke-width="2"
												stroke="currentColor"
												class="size-4"
											>
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="m14.25 6.75-4.5 4.5 4.5 4.5"
												/>
											</svg>
										</button>
									{/if}
									<button
										class="p-2 rounded-full border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
										on:click={closeAnalysisPanel}
										aria-label={$i18n.t('Close')}
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											fill="none"
											viewBox="0 0 24 24"
											stroke-width="2"
											stroke="currentColor"
											class="size-4"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												d="M6 18 18 6M6 6l12 12"
											/>
										</svg>
									</button>
								</div>
							</div>
						</div>

						<div class="p-4 overflow-y-auto flex-1 space-y-3">
							{#if analysisLoading && !analysisResult}
								<!-- Full loading state when no data yet -->
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
												<div class="text-sm font-semibold truncate">{analysisPrimaryStatus}</div>
												<div class="text-xs text-gray-500 dark:text-gray-400">
													{analysisSecondaryStatus}
												</div>
											</div>
											<div
												class="text-[11px] font-semibold text-gray-500 dark:text-gray-400 pl-3 border-l border-gray-200 dark:border-gray-800"
											>
												{analysisStage === 'layer2' || analysisStage === 'complete' ? '2/2' : '1/2'}
											</div>
										</div>

										<div
											class="h-2 w-full rounded-full bg-gray-200/80 dark:bg-gray-800 overflow-hidden"
										>
											<div
												class="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all duration-300 ease-out"
												style={`width: ${analysisLoadingPercent}%`}
											></div>
										</div>

										<div class="grid grid-cols-2 gap-3 text-xs text-gray-600 dark:text-gray-300">
											<div class="flex items-center gap-2">
												<div
													class={`size-2.5 rounded-full ${
														analysisStage === 'layer1'
															? 'bg-blue-500 animate-pulse'
															: 'bg-gray-300 dark:bg-gray-700'
													}`}
												></div>
												<div class="truncate">{$i18n.t('Analyzing reasoning structure...')}</div>
											</div>
											<div class="flex items-center gap-2">
												<div
													class={`size-2.5 rounded-full ${
														analysisStage === 'layer2'
															? 'bg-indigo-500 animate-pulse'
															: 'bg-gray-300 dark:bg-gray-700'
													}`}
												></div>
												<div class="truncate">
													{analysisStage === 'layer2'
														? `${analysisLayer2Progress.completed}/${analysisLayer2Progress.total || '?'} ${$i18n.t('nodes')}`
														: $i18n.t('Analyzing details...')}
												</div>
											</div>
										</div>
									</div>
								</div>
							{:else if analysisError}
								<div
									class="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/40 text-red-700 dark:text-red-200 p-3 text-sm"
								>
									{analysisError}
								</div>
							{:else if analysisResult}
								<!-- Show progressive loading banner when still loading but have partial results -->
								{#if analysisLoading && analysisStage === 'layer2'}
									<div
										class="rounded-lg border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/30 p-3 mb-3 flex items-center gap-3"
									>
										<Spinner className="size-4 text-blue-600 dark:text-blue-400" />
										<div class="text-sm text-blue-700 dark:text-blue-300">
											<span class="font-medium">{$i18n.t('Analyzing details...')}</span>
											<span class="text-blue-600 dark:text-blue-400 ml-2">
												{analysisLayer2Progress.completed}/{analysisLayer2Progress.total}
												{$i18n.t('nodes')}
											</span>
										</div>
									</div>
								{/if}
								<div class="space-y-3 text-sm">
									<SvelteFlowProvider>
										<ReasoningTree
											nodes={analysisResult.nodes ?? []}
											edges={analysisResult.edges ?? []}
											sections={analysisResult.sections ?? []}
											overthinkingAnalysis={errorDetectionResult?.overthinking_analysis ?? null}
											detectedErrors={errorDetectionResult?.errors ?? []}
											{chatId}
											messageId={message.id}
											model={analysisModel}
											{analysisStage}
											on:highlight={(e) => {
												highlightReasoningSentence(
													e.detail?.sentence,
													e.detail?.sectionStart,
													e.detail?.sectionEnd,
													analysisResult?.sections ?? [],
													e.detail?.isError,
													e.detail?.sectionNumbers,
													e.detail?.highlightType
												);
											}}
										/>
									</SvelteFlowProvider>
								</div>
							{:else}
								<div class="text-sm text-gray-600 dark:text-gray-300">
									{$i18n.t('No analysis available')}
								</div>
							{/if}
						</div>
					{/if}
				</div>
			</div>
		</div>
	{/if}
{/key}

<style>
	.buttons::-webkit-scrollbar {
		display: none; /* for Chrome, Safari and Opera */
	}

	.buttons {
		-ms-overflow-style: none; /* IE and Edge */
		scrollbar-width: none; /* Firefox */
	}

	/* Section highlight style - transparent yellow glass effect */
	/* Use :global() because these marks are created dynamically via JavaScript */
	:global(mark.reasoning-highlight) {
		background: linear-gradient(
			135deg,
			rgba(251, 191, 36, 0.15) 0%,
			rgba(253, 224, 71, 0.2) 100%
		) !important;
		border-radius: 4px;
		padding: 2px 4px;
		box-shadow: inset 0 0 0 1px rgba(251, 191, 36, 0.25);
		transition: all 0.3s ease-in-out;
		color: inherit;
	}

	:global(.dark mark.reasoning-highlight) {
		background: linear-gradient(
			135deg,
			rgba(251, 191, 36, 0.12) 0%,
			rgba(253, 224, 71, 0.18) 100%
		) !important;
		box-shadow: inset 0 0 0 1px rgba(251, 191, 36, 0.2);
	}

	/* Error highlight style - transparent red glass effect */
	:global(mark.reasoning-error-highlight) {
		background: linear-gradient(
			135deg,
			rgba(239, 68, 68, 0.12) 0%,
			rgba(248, 113, 113, 0.18) 100%
		) !important;
		border-radius: 4px;
		padding: 2px 4px;
		box-shadow: inset 0 0 0 1px rgba(239, 68, 68, 0.3);
		transition: all 0.3s ease-in-out;
		text-decoration: underline wavy rgba(220, 38, 38, 0.5);
		text-underline-offset: 3px;
		color: inherit;
	}

	:global(.dark mark.reasoning-error-highlight) {
		background: linear-gradient(
			135deg,
			rgba(239, 68, 68, 0.15) 0%,
			rgba(248, 113, 113, 0.2) 100%
		) !important;
		box-shadow: inset 0 0 0 1px rgba(248, 113, 113, 0.35);
		text-decoration: underline wavy rgba(248, 113, 113, 0.6);
	}

	/* Answer highlight style - transparent purple glass effect for overthinking answer positions */
	:global(mark.reasoning-answer-highlight) {
		background: linear-gradient(
			135deg,
			rgba(139, 92, 246, 0.12) 0%,
			rgba(167, 139, 250, 0.18) 100%
		) !important;
		border-radius: 4px;
		padding: 2px 4px;
		box-shadow: inset 0 0 0 1px rgba(139, 92, 246, 0.3);
		transition: all 0.3s ease-in-out;
		border-bottom: 2px solid rgba(139, 92, 246, 0.5);
		color: inherit;
	}

	:global(.dark mark.reasoning-answer-highlight) {
		background: linear-gradient(
			135deg,
			rgba(139, 92, 246, 0.15) 0%,
			rgba(167, 139, 250, 0.22) 100%
		) !important;
		box-shadow: inset 0 0 0 1px rgba(167, 139, 250, 0.4);
		border-bottom: 2px solid rgba(167, 139, 250, 0.6);
	}
</style>
