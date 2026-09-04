const groq = String.raw;

export const overviewQuery = groq`{
	"interviewCount": count(*[_type == "interview"]),
	"literatureCount": count(*[_type == "literatureItem"]),
	"miviludesCriteriaCount": count(*[_type == "miviludesCriterion"]),
	"interviews": *[_type == "interview"]{
		batch, language, method, translated, totalWordCount
	}
}`;

export const interviewListQuery = groq`*[_type == "interview"] | order(dateTime asc){
	corpusId, batch, dateTime, language, method, translated, location, totalWordCount
}`;

export const literatureListQuery = groq`*[_type == "literatureItem"] | order(year desc, title asc){
	corpusId, title, authors, year, type, source, language, tags
}`;

export const dictionaryListQuery = groq`*[_type == "dictionaryEntry"] | order(term asc){
	corpusId, term, source, language, citation
}`;

export const customTermListQuery = groq`*[_type == "customTerm"] | order(term asc){
	corpusId, term, definition, relatedTerms
}`;

export const interviewDetailQuery = groq`*[_type == "interview" && corpusId == $corpusId][0]{
	corpusId, batch, dateTime, dateTimePrecision, method, language, translated,
	translationLanguage, location, age, gender, nationality, mainLanguage,
	languageSpoken, nQuestions, nAnswers, totalWordCount, text, translationText
}`;

export const literatureDetailQuery = groq`*[_type == "literatureItem" && corpusId == $corpusId][0]{
	corpusId, title, authors, year, type, source, doi, isbn, language, tags,
	rawFile, dateAdded, abstract, notes
}`;

export const miviludesCriteriaListQuery = groq`*[_type == "miviludesCriterion"] | order(order asc){
	corpusId, criterionFr, criterionEn, order, source, citation
}`;
