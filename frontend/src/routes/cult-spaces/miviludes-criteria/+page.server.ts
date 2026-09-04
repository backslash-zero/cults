import { sanity } from '$lib/server/sanity';
import { miviludesCriteriaListQuery } from '$lib/server/queries';
import type { MiviludesCriterion } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const criteria = await sanity.fetch<MiviludesCriterion[]>(miviludesCriteriaListQuery);
	return { criteria };
};
