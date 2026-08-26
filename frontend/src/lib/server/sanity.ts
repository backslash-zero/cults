import { createClient } from '@sanity/client';
import { env } from '$env/dynamic/private';

const projectId = env.SANITY_PROJECT_ID || 'dm4p8gdv';
const dataset = env.SANITY_DATASET || 'production';
const apiVersion = env.SANITY_API_VERSION || '2026-01-01';
const token = env.SANITY_API_READ_TOKEN || undefined;

export const sanity = createClient({
	projectId,
	dataset,
	apiVersion,
	token,
	useCdn: !token,
	perspective: 'published'
});
