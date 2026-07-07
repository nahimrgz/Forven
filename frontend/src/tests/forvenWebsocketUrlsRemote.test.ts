import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const REMOTE_ORIGIN = 'https://3dm7hr0p-5173.usw3.devtunnels.ms';

// Simulate the browser being served through a VSCode dev tunnel: core resolves an
// absolute same-origin base (the Vite `/api` proxy), never a direct `:8003` port.
// NOTE: the factory is hoisted above every top-level binding, so the origin is
// inlined here rather than referencing REMOTE_ORIGIN.
vi.mock('../lib/api/core', () => ({
	ACTIVE_API_BASE: 'https://3dm7hr0p-5173.usw3.devtunnels.ms/api',
	API_BASE: 'https://3dm7hr0p-5173.usw3.devtunnels.ms/api',
	isLocalHost: (hostname: string) =>
		['localhost', '127.0.0.1', '::1', ''].includes((hostname || '').toLowerCase()),
	fetchApi: vi.fn()
}));

import { getForvenLiveWebSocketUrls } from '../lib/api/forven';

describe('Forven websocket URLs behind a dev tunnel', () => {
	let originalLocation: Location;

	beforeEach(() => {
		originalLocation = window.location;
		Object.defineProperty(window, 'location', {
			configurable: true,
			value: new URL(`${REMOTE_ORIGIN}/`) as unknown as Location
		});
	});

	afterEach(() => {
		Object.defineProperty(window, 'location', {
			configurable: true,
			value: originalLocation
		});
	});

	it('uses the same-origin proxy and never targets :8003', () => {
		const urls = getForvenLiveWebSocketUrls();

		expect(urls[0]).toBe('wss://3dm7hr0p-5173.usw3.devtunnels.ms/api/ws/live');
		expect(urls.every((url) => !url.includes(':8003'))).toBe(true);
		expect(urls.every((url) => url.startsWith('wss://3dm7hr0p-5173.usw3.devtunnels.ms/'))).toBe(
			true
		);
	});
});
