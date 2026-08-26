import {defineField, defineType, defineArrayMember} from 'sanity'
import {TagIcon} from '@sanity/icons'

export const customTerm = defineType({
  name: 'customTerm',
  title: 'Custom term',
  type: 'document',
  icon: TagIcon,
  fields: [
    defineField({
      name: 'corpusId',
      title: 'Corpus ID',
      type: 'string',
      description: 'The stable corpus/custom_terms ID, e.g. "term-semantic-drift". Used to upsert from the file-based corpus — never used as the Sanity _id.',
      validation: (rule) => rule.required(),
    }),
    defineField({name: 'term', title: 'Term', type: 'string', validation: (rule) => rule.required()}),
    defineField({name: 'definition', title: 'Definition', type: 'text'}),
    defineField({
      name: 'relatedTerms',
      title: 'Related corpus IDs',
      type: 'array',
      of: [defineArrayMember({type: 'string'})],
      description: 'Plain corpusId strings (lit-/dict-/term-/interview IDs) this term relates to — soft links, not resolved Sanity references.',
    }),
    defineField({name: 'dateAdded', title: 'Date added', type: 'date'}),
    defineField({name: 'notes', title: 'Notes', type: 'text'}),
  ],
  preview: {
    select: {title: 'term', subtitle: 'corpusId'},
  },
})
