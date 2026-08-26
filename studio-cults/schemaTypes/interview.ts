import {defineField, defineType} from 'sanity'
import {CommentIcon} from '@sanity/icons'

export const interview = defineType({
  name: 'interview',
  title: 'Interview',
  type: 'document',
  icon: CommentIcon,
  fields: [
    defineField({
      name: 'corpusId',
      title: 'Corpus ID',
      type: 'string',
      description: 'The stable corpus/interviews ID, e.g. "b1-aug05-1650" or "ig-02". Used to upsert from the file-based corpus — never used as the Sanity _id.',
      validation: (rule) => rule.required(),
    }),
    defineField({name: 'batch', title: 'Batch', type: 'string'}),
    defineField({name: 'dateTime', title: 'Date/time', type: 'string'}),
    defineField({name: 'dateTimePrecision', title: 'Date/time precision note', type: 'string'}),
    defineField({name: 'method', title: 'Method', type: 'string'}),
    defineField({name: 'language', title: 'Language', type: 'string'}),
    defineField({name: 'translated', title: 'Translated', type: 'boolean'}),
    defineField({name: 'translationLanguage', title: 'Translation language', type: 'string'}),
    defineField({name: 'location', title: 'Location', type: 'string'}),
    defineField({name: 'age', title: 'Interviewee age', type: 'string'}),
    defineField({name: 'gender', title: 'Interviewee gender', type: 'string'}),
    defineField({name: 'nationality', title: 'Interviewee nationality', type: 'string'}),
    defineField({name: 'mainLanguage', title: 'Interviewee main language', type: 'string'}),
    defineField({name: 'languageSpoken', title: 'Language spoken in interview', type: 'string'}),
    defineField({name: 'nQuestions', title: 'Number of questions', type: 'number'}),
    defineField({name: 'nAnswers', title: 'Number of answers', type: 'number'}),
    defineField({name: 'totalWordCount', title: 'Total word count', type: 'number'}),
    defineField({name: 'text', title: 'Transcript text', type: 'text'}),
    defineField({name: 'translationText', title: 'Translation text', type: 'text'}),
  ],
  preview: {
    select: {title: 'corpusId', subtitle: 'dateTime'},
  },
})
