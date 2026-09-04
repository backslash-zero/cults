import {defineField, defineType} from 'sanity'
import {WarningOutlineIcon} from '@sanity/icons'

export const miviludesCriterion = defineType({
  name: 'miviludesCriterion',
  title: 'MIVILUDES criterion',
  type: 'document',
  icon: WarningOutlineIcon,
  fields: [
    defineField({
      name: 'corpusId',
      title: 'Corpus ID',
      type: 'string',
      description: 'The stable corpus/metadata/miviludes_criteria ID, e.g. "crit-mental-destabilization". Used to upsert from the file-based corpus — never used as the Sanity _id.',
      validation: (rule) => rule.required(),
    }),
    defineField({name: 'criterionFr', title: 'Criterion (FR)', type: 'text', validation: (rule) => rule.required()}),
    defineField({name: 'criterionEn', title: 'Criterion (EN)', type: 'text', validation: (rule) => rule.required()}),
    defineField({name: 'order', title: 'Order', type: 'number'}),
    defineField({name: 'source', title: 'Source', type: 'string'}),
    defineField({name: 'citation', title: 'Citation', type: 'string'}),
    defineField({name: 'dateAdded', title: 'Date added', type: 'date'}),
  ],
  preview: {
    select: {title: 'corpusId', subtitle: 'source'},
  },
})
