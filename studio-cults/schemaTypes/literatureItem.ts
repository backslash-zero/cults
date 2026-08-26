import {defineField, defineType, defineArrayMember} from 'sanity'
import {DocumentTextIcon} from '@sanity/icons'

export const literatureItem = defineType({
  name: 'literatureItem',
  title: 'Literature item',
  type: 'document',
  icon: DocumentTextIcon,
  fields: [
    defineField({
      name: 'corpusId',
      title: 'Corpus ID',
      type: 'string',
      description: 'The stable corpus/literature ID, e.g. "lit-durkheim1912-elementary-forms". Used to upsert from the file-based corpus — never used as the Sanity _id.',
      validation: (rule) => rule.required(),
    }),
    defineField({name: 'title', title: 'Title', type: 'string', validation: (rule) => rule.required()}),
    defineField({
      name: 'authors',
      title: 'Authors',
      type: 'array',
      of: [defineArrayMember({type: 'string'})],
    }),
    defineField({name: 'year', title: 'Year', type: 'string'}),
    defineField({
      name: 'type',
      title: 'Type',
      type: 'string',
      options: {
        list: [
          {title: 'Article', value: 'article'},
          {title: 'Book chapter', value: 'book_chapter'},
          {title: 'Book', value: 'book'},
          {title: 'Report', value: 'report'},
          {title: 'Other', value: 'other'},
        ],
        layout: 'radio',
      },
    }),
    defineField({name: 'source', title: 'Source (journal/publisher)', type: 'string'}),
    defineField({name: 'language', title: 'Language', type: 'string'}),
    defineField({
      name: 'tags',
      title: 'Tags',
      type: 'array',
      of: [defineArrayMember({type: 'string'})],
    }),
    defineField({name: 'rawFile', title: 'Raw file path', type: 'string'}),
    defineField({name: 'dateAdded', title: 'Date added', type: 'date'}),
    defineField({name: 'notes', title: 'Notes', type: 'text'}),
  ],
  preview: {
    select: {title: 'title', subtitle: 'corpusId'},
  },
})
