import type { MiviludesDocument } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ fetch }) => {
	const documents: MiviludesDocument[] = await (await fetch('/data/miviludes-documents.json')).json();
	return { documents };
};
