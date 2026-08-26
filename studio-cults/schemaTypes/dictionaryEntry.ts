import {defineField, defineType} from 'sanity'
import {BookIcon} from '@sanity/icons'

export const dictionaryEntry = defineType({
  name: 'dictionaryEntry',
  title: 'Dictionary entry',
  type: 'document',
  icon: BookIcon,
  fields: [
    defineField({
      name: 'corpusId',
      title: 'Corpus ID',
      type: 'string',
      description: 'The stable corpus/dictionaries ID, e.g. "dict-thesoz-sekte". Used to upsert from the file-based corpus — never used as the Sanity _id.',
      validation: (rule) => rule.required(),
    }),
    defineField({name: 'source', title: 'Source', type: 'string', description: 'e.g. THESOZ, SAGE Encyclopedia...'}),
    defineField({name: 'term', title: 'Term', type: 'string', validation: (rule) => rule.required()}),
    defineField({name: 'language', title: 'Language', type: 'string'}),
    defineField({
      name: 'definition',
      title: 'Definition',
      type: 'text',
      description: 'Quoted verbatim from the source — never paraphrased.',
    }),
    defineField({name: 'citation', title: 'Citation', type: 'string'}),
    defineField({name: 'dateAdded', title: 'Date added', type: 'date'}),
  ],
  preview: {
    select: {title: 'term', subtitle: 'source'},
  },
})
