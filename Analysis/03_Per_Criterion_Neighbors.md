# Per-criterion nearest neighbours — full listing

Data inventory: [`00_Available_Data.md`](00_Available_Data.md). Index: [`Results_Draft.md`](Results_Draft.md).
Summary table + methodology: [`02_Corpus_Centroids.md`](02_Corpus_Centroids.md).

For each of the 17 sectarian-drift criteria individually — each criterion is its own query
vector, not a group centroid — its nearest neighbours in three separate pools, each pool's own
points only: 5 nearest named entities, 10 nearest literature expressions, 10 nearest interview
segments. Machine-readable version (all 3 pools, same counts):
[`data/criterion_nearest_neighbors.csv`](data/criterion_nearest_neighbors.csv) (425 rows).

---

## Coverage-gap analysis: which criteria are well- vs. poorly-covered, and where

Quantifies what the per-criterion listings below only let you eyeball: for each criterion,
mean cosine similarity of its nearest neighbours, per pool (5 entities, 10 literature, 10
interview segments) — a direct number for "how conceptually close is the *closest* material
this corpus has to this official criterion," not just which labels come up. Higher = a richer,
more specific vocabulary for that concept exists somewhere in the corpus; lower = even the
best match is a loose, generic one.

| Criterion | Literature | Entities | Interviews | Avg |
|---|---|---|---|---|
| Radical behavior change | 0.662 | 0.613 | 0.551 | 0.609 |
| Authoritarian/opaque group | 0.651 | 0.525 | 0.595 | 0.590 |
| Difficulty leaving the group | 0.649 | 0.512 | 0.605 | 0.589 |
| Mental destabilization | 0.646 | 0.494 | 0.552 | 0.564 |
| Indoctrination of children | 0.567 | 0.567 | 0.544 | 0.559 |
| Infiltration of public authorities | 0.585 | 0.545 | 0.547 | 0.559 |
| Deceptive recruitment | 0.628 | 0.543 | 0.506 | 0.559 |
| Rupture with origin environment | 0.585 | 0.533 | 0.532 | 0.550 |
| Alarming dietary change | 0.582 | 0.520 | 0.544 | 0.549 |
| Harsh living conditions | 0.585 | 0.500 | 0.528 | 0.538 |
| Physical-integrity harm | 0.596 | 0.478 | 0.532 | 0.535 |
| Contesting public order | 0.564 | 0.517 | 0.517 | 0.533 |
| Rejection of outside world | 0.581 | 0.466 | 0.517 | 0.521 |
| Financial demands/opacity | 0.557 | 0.470 | 0.523 | 0.517 |
| Dubious/exclusive care | 0.567 | 0.484 | 0.495 | 0.515 |
| Violation of Republic principles | 0.539 | 0.525 | 0.451 | 0.505 |
| **Legal disputes** | **0.493** | **0.447** | **0.418** | **0.453** |

**Best covered**: *Radical behavior change*, *Authoritarian/opaque group*, *Difficulty leaving
the group*, *Mental destabilization* — all sit near or above 0.65 in literature specifically,
reflecting decades of psychological/sociological NRM research with dedicated vocabulary
(thought reform, charismatic authority, self-transformation).

**Worst covered, consistently across all three pools — a real, triangulated gap, not a
literature-specific artifact**: **Legal disputes** is the single weakest criterion in *every*
pool (literature 0.493, entities 0.447, interviews 0.418 — all last place). **Violation of
Republic principles** is second-weakest overall (0.505) and the weakest in interviews
specifically (0.451) — unsurprising, since "the Republic" is a French constitutional concept
with no direct equivalent in the mostly-English, mostly-Anglo-American academic literature or
in ordinary interviewee speech. Both are administrative/legal/French-regulatory-specific
criteria without a close counterpart in English-language NRM scholarship or lay discourse —
consistent with the "MIVILUDES report vs. academic literature register" divergence in the main
findings: these two criteria sit closer to French legal/administrative language than to
anything this corpus otherwise discusses.

---

### Mental destabilization

*La déstabilisation mentale et la sujétion psychologique ou physique conduisant à des actions ou abstentions gravement préjudiciables aux personnes et plus généralement à la perte d'esprit critique et d'autonomie*

**English:** Mental destabilization and psychological or physical subjugation leading to actions or omissions gravely detrimental to individuals, and more generally to the loss of critical thinking and autonomy

**Nearest entities:**

1. thought reform and the psychology of totalism
2. dianetics: the modern science of mental health
3. diagnostic and statistical manual of mental disorders
4. submissiveness
5. conscientious objection

**Nearest literature expressions:**

1. Groups aiming through maneuvers of psychological destabilization to obtain unconditional allegiance from their followers, a diminution in critical thinking, and a rupture with commonly accepted references (ethical, scientific, civic, educational), and entailing dangers for individual freedoms, health, education, and democratic institutions.
2. It is a deviation from freedom of thought, opinion or religion which undermines public order, laws or regulations, or persons' fundamental rights, security or integrity.
3. the goal or effect to create or to exploit the state of mental or physical dependence of people who are participating in its activities
4. Destabilize a person's sense of self.
5. a useful though scientifically imprecise concept which refers to an array of complex phenomena resulting in the impairment of the individual's cognitive and social functioning
6. a useful though scientifically imprecise concept which refers to an array of complex phenomena resulting in the impairment of the individual's cognitive and social functioning
7. brainwashing consists of overwhelming or irresistible 'extrinsic' influence to which the inner qualities of the person are irrelevant
8. mental manipulation
9. mental manipulation
10. mental manipulation

**Nearest interview segments:**

1. So, I'd picture something American, okay — brainwashing, suppressive, abusive.
2. I would say it's people participating in something, and then there's a person who has another agenda and kind of manipulates them, and uses, like — their, I don't know — participation, or — I think that's not the right word — but, like, their willingness, and, like, their good [nature] to participate in something.
3. It means committing to rules, committing to maybe a financial engagement, and it comes with giving up some individual freedoms, let's say.
4. Something suspicious, something dangerous, something not normal.
5. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
6. A priori, dans la définition même, ça implique un truc négatif.
7. But I suppose most commonly I'd consider groups of people with some commonly considered 'alternative' beliefs and a person, figure or certain idea as a central element within their communal belief system.
8. I guess there's a definition of cult, and hopefully they're assessed based on some definition, but I guess a cult is anything that becomes too extreme in an ideology, and kind of just messes with the people themselves.
9. I think the first thing that comes to my mind is more, probably, the bad side of it — the bad side of it.
10. something nasty, something really nasty.

### Rupture with origin environment

*La rupture avec l'environnement d'origine (proches, famille)*

**English:** Rupture with the person's environment of origin (close relations, family)

**Nearest entities:**

1. radical departures
2. self transformation
3. falling from the faith
4. concerned relatives
5. the family (formerly the children of god)

**Nearest literature expressions:**

1. religious bodies in a relatively high state of tension with their environments
2. individual disinvolvement (private distancing, limiting emotional investment)
3. a radical departure
4. exit counseling
5. exit counselling
6. Destabilize a person's sense of self.
7. Catastrophic millennialists who live in isolated communities are inherently in tension with mainstream society.
8. Movements that encourage dissociation from the (biological) family and entry into an alternative community structure tend to be termed 'cults.'
9. personal transformation may be symbolic and/or social
10. disengagement/defection

**Nearest interview segments:**

1. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
2. Because they are an enclosed group with one strange thing in the full center of their attention (for each of the new age groups)
3. Something suspicious, something dangerous, something not normal.
4. something nasty, something really nasty.
5. Une communauté.
6. So, yeah, kind of those stories of people trying to escape those cults.
7. Um — "Missbrauch" [German for "abuse"/"misuse"].
8. So it's mostly, like, you have a negative association with.
9. But there was an extreme part of the religion where the people who grew up there — like, if your parents were there, you grew up there, and you don't know anything else than that, so anything outside of that is wrong.
10. Yeah, because it's like, once you start, it's hard to leave.

### Radical behavior change

*Le changement radical de comportement*

**English:** Radical change in behavior

**Nearest entities:**

1. self transformation
2. human individual metamorphosis
3. self-transformation
4. the consciousness reformation
5. radical departures

**Nearest literature expressions:**

1. sudden personality change
2. a radical departure
3. radical departures
4. Get the person to drastically reinterpret his or her life's history and radically alter his or her worldview and accept a new version of reality and causality.
5. radical patterns of social organization
6. the reality of the transformation they experience is both far more immediate and far more complex
7. thought reform
8. thought reform
9. transformational leader
10. The role of women in society is undergoing profound change

**Nearest interview segments:**

1. I think it changes depending on what I've been doing in the last days.
2. Okay, do you have a specific example that would come to your mind immediately?
3. Un groupe qui suit une doctrine.
4. Religion.
5. Religion.
6. That's interesting.
7. Qui suit des règles.
8. Something suspicious, something dangerous, something not normal.
9. something nasty, something really nasty.
10. This comes to my mind.

### Rejection of outside world

*Le refus de l'autre et le dénigrement du monde extérieur : l'absence totale d'accès aux médias ou moyens de communication*

**English:** Rejection of others and denigration of the outside world: total absence of access to media or means of communication

**Nearest entities:**

1. alienation and charisma
2. submissiveness
3. silence
4. self-religion
5. thought reform and the psychology of totalism

**Nearest literature expressions:**

1. world rejecting
2. apocalyptic beliefs, or at least world-rejecting beliefs
3. We had to disappear in order to be isolated. In other words, we had to be separated out, or lifted out, in order to be free to do what we had to do.
4. brainwashing consists of overwhelming or irresistible 'extrinsic' influence to which the inner qualities of the person are irrelevant
5. They reject the values of society that they deem to be ungodly.
6. We don't approach/go towards the groups, we do not have the time.
7. Moral evil is widely understood to result from ignorance or lack of awareness of the true nature of the Self.
8. There is no dogma, no orthodoxy, and essentially, no agreement on where the boundaries of the movement are and who is or is not part of it.
9. The simplest form of secrecy merely implies that two people share information unknown to anybody else.
10. The current planetary situation is irreversibly escaping all human control . . . All creative and positive forces are strangled . . . we refuse to participate in the assassination of our carrier the Earth, we leave this world where our voices can no longer be heard

**Nearest interview segments:**

1. I would say, okay, they also practice the ideology so strictly that they kind of raise their people to not know anything else than that, and they're also not open to the world.
2. Cult, for me, also means a lack of freedom.
3. So, if I understand well, the main feature for you is this — opposing something?
4. So it's mostly, like, you have a negative association with.
5. And, yeah, so they are very isolated, in a way — they need to follow strict principles, there are a lot of rules, and very little contact is allowed outside of the group.
6. C'est quelque chose qu'on évite de fréquenter, en général, parce que, dans mon style de vie...
7. They're not really giving you something, they just pretend to give you something, and then you, being part of the cult  are not aware of that.
8. Oh, I — yeah, I don't have any names, or — yeah, I don't remember any names.
9. It's a group, but without any kind of hierarchy — key members, or a set number of members.
10. I think the fact that we get obsessed with it — if it's a very enclosed, non-welcoming community, it could make it feel like that — but I don't think there's a leader that we follow.

### Harsh living conditions

*Les conditions de vie particulièrement éprouvantes ou déstabilisantes*

**English:** Particularly harsh or destabilizing living conditions

**Nearest entities:**

1. intensified conflict
2. life
3. coercive persuasion
4. concerned relatives
5. conscientious objection

**Nearest literature expressions:**

1. religious bodies in a relatively high state of tension with their environments
2. People who lacked a strong sense of boundaries, or conversely, were too rigid and set in their ways, were deemed unsuitable.
3. contemporary cases of extreme violence
4. Coercive or involuntary imposition of a defective or false worldview
5. Destabilize a person's sense of self.
6. a very controlling environment existed in both as well.
7. alternating periods of severity and leniency
8. an extreme, extrinsic brainwashing model
9. the dire economic conditions and abject poverty in many parts of the Northern Caucasus and elsewhere in Russia can be a powerful radicalizing factor
10. the exercise of serious and repeated pressures or techniques aimed at altering the capacity of judgment

**Nearest interview segments:**

1. something nasty, something really nasty.
2. Something suspicious, something dangerous, something not normal.
3. C'est pas forcément très bien.
4. So, I'd picture something American, okay — brainwashing, suppressive, abusive.
5. I feel like usually the leaders are super abusive.
6. But, you know, here's where I see the privilege — I believe, in some areas of the world you are dominated by the conditions around you, the daily struggles in life, where people have to be in fight-or-flight mode, so they don't have the privilege to think or rethink their beliefs and say "oh, this doesn't fit me, I'd like to follow this cult or that belief" — while in some areas of the world, especially in war zones, when people are very close to death, I believe they'd rather stick to what they were raised on, because they believe this is the remedy, or this is what gives them the better life they'll have after.
7. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
8. Any particular example?
9. And the members have to live by the rules of the group — very strict rules.
10. Okay, no — that can be — this goes more into the religious and against a particular topic, but in a super extreme way.

### Deceptive recruitment

*Les méthodes de recrutement trompeuses*

**English:** Deceptive recruitment methods

**Nearest entities:**

1. coercive persuasion
2. a social psychological critique of 'brainwashing' claims about recruitment to new religions
3. the training of the cadre
4. doctrine over person
5. media trickery

**Nearest literature expressions:**

1. If recruitment techniques are so sinister,' asks Levine, 'why do they so rarely work?' ( 1984B: 27).
2. brainwashing has something to do with recruitment mechanisms
3. The first phase of recruitment is evidently one of seduction.
4. Cult recruiters are alleged to systematically employ hypnotists' techniques
5. Cult recruiters are alleged to systematically employ hypnotists' techniques
6. coercive persuasion
7. coercive persuasion
8. cult recruitment
9. This is one reason why the subject of a cult recruitment attempt should not be viewed as a passive target overwhelmed by coercively compelling 'mind control tactics'
10. The brainwashing hypothesis is predicated on the loss of a recruit's ability to make informed decisions

**Nearest interview segments:**

1. It's really easy to get kind of convinced to join a cult, by using some kind of false induction, false information.
2. But, yeah, it's a system of belief that spreads and exploits the gullible.
3. des groupes fermés qui endoctrinent des gens en échange d'argent, la plupart du temps, et qui peuvent conduire à des dérives dangereuses.
4. Yeah, so, yeah, a very different way of getting raised, I guess.
5. Somehow they try to convince people to join the cult.
6. but I might be wrong.
7. exploitation?
8. En les attirant vers quelque chose.
9. Manipulation, groupe, idéologie.
10. So there's a lot of brainwashing and abuse allegations.

### Indoctrination of children

*L'embrigadement des enfants*

**English:** The indoctrination of children

**Nearest entities:**

1. children in new religions
2. let our children go!
3. the training of the cadre
4. demonstration
5. the children of god

**Nearest literature expressions:**

1. The children are our future.
2. child abuse
3. New Age children
4. a middle position
5. He is 12 or 13 years old.
6. The children are made to take part in Satanic rituals
7. Love Our Children
8. child custody cases
9. the imitating multitude
10. The Children of God

**Nearest interview segments:**

1. En les attirant vers quelque chose.
2. Des groupements...
3. Une communauté.
4. Illuminati.
5. Weird people coming together to do weird things.
6. You know, like, these people have — I don't know — thousands of children, because it's [encouraged] by nature.
7. ça peut être large.
8. Qui suit des règles.
9. Groupe.
10. People dressed in white

### Authoritarian/opaque group

*L'existence d'un groupe organisé sur un mode autoritaire, opaque et cloisonné, avec présence d'un dirigeant de type leader charismatique ou praticien référent exclusif*

**English:** The existence of a group organized in an authoritarian, opaque, and compartmentalized manner, with a charismatic-leader-type figure or an exclusive referent practitioner at its head

**Nearest entities:**

1. charismatic authority
2. charismatic leadership
3. evolving perspectives on charismatic leadership
4. satanic groups
5. unifi cationism

**Nearest literature expressions:**

1. Charismatic authority: Leadership was secretive and inaccessible.
2. apocalyptic worldview, charismatic leadership, high levels of internal control, and intense internal solidarity that produces isolation from the surrounding society
3. Authoritarian submission, involving dependency on and deference to strong leaders.
4. Leaders are often described as guides, trainers, counselors who, formally at least, typically have more limited authority than counterparts in world-rejecting/transformative movements.
5. a dominant male charismatic leader, a hierarchical male leadership structure, doctrinal beliefs that privilege male superiority, strong gender norms governing behavior, and a highly paternalistic family structure
6. It must be noted that charismatic leadership is socially constructed, and if there are no believers in a person's claimed access to an unseen source of authority, then there is no charismatic leader.
7. movements with distinctly apocalyptic worldviews, charismatic leadership, tight internal social control, and intense communal solidarity
8. They frequently refer to the movements as malevolent and destructive cults or totalistic groups that demand hyper-compliance and use coercive power and undue authority.
9. Evangelism, or concern with proselytization (there may be exceptions here in terms of groups which have isolated themselves in such a way as to inhibit outreach-such groups may be particularly volatile).
10. They frequently refer to the movements as malevolent and destructive cults or totalistic groups that demand hypercompliance and use coercive power and undue authority.

**Nearest interview segments:**

1. Because they have a singular leader, it's not easy to get into the group, you have to go through an integration process, and it's a very closed community.
2. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
3. It's a group, but without any kind of hierarchy — key members, or a set number of members.
4. And, yeah, so they are very isolated, in a way — they need to follow strict principles, there are a lot of rules, and very little contact is allowed outside of the group.
5. All those smaller church denominations with a charismatic leader and their own little special lore — something that usually exploits vulnerable people, gullible people, you know — extracts resources and sex from them, depending on the type of cult.
6. des groupes fermés qui endoctrinent des gens en échange d'argent, la plupart du temps, et qui peuvent conduire à des dérives dangereuses.
7. I think the fact that we get obsessed with it — if it's a very enclosed, non-welcoming community, it could make it feel like that — but I don't think there's a leader that we follow.
8. A group of people that are blindly devoted, or blindly follow a belief without questioning it.
9. These would happen on sight but also kinda behind the doors and there was also several groups with different leaders.
10. But I suppose most commonly I'd consider groups of people with some commonly considered 'alternative' beliefs and a person, figure or certain idea as a central element within their communal belief system.

### Difficulty leaving the group

*De grandes difficultés voire une impossibilité pour un membre de quitter ledit groupe*

**English:** Great difficulty, or even the impossibility, for a member to leave the group

**Nearest entities:**

1. radical departures
2. satanic groups
3. concerned relatives
4. away team
5. older members

**Nearest literature expressions:**

1. it is almost impossible for the person to leave the group. The submission is complete.
2. it is almost impossible for him to leave the group .... The submission is complete.
3. a lengthy indoctrination by the group is frequently necessary
4. a term that refers to controversial groups that are odd and even dangerous to group members and others
5. high-level defections may traumatize the leader and enhance group volatility
6. the internal volatility of the group magnified the threat
7. As hard as it might be to fathom belonging to a cult
8. it is not possible to say a priori whether a group is or is not dangerous
9. few were aware of a steady stream of disaffected members exiting these movements by the back door
10. I suffered from the group pressure and anxieties

**Nearest interview segments:**

1. And the members have to live by the rules of the group — very strict rules.
2. Because they have a singular leader, it's not easy to get into the group, you have to go through an integration process, and it's a very closed community.
3. Yeah, because it's like, once you start, it's hard to leave.
4. And, yeah, so they are very isolated, in a way — they need to follow strict principles, there are a lot of rules, and very little contact is allowed outside of the group.
5. des groupes fermés qui endoctrinent des gens en échange d'argent, la plupart du temps, et qui peuvent conduire à des dérives dangereuses.
6. And, top — like, not everybody can be there, and, like, kind of a higher status.
7. These would happen on sight but also kinda behind the doors and there was also several groups with different leaders.
8. Something suspicious, something dangerous, something not normal.
9. I feel like usually the leaders are super abusive.
10. Un groupe qui suit une doctrine.

### Physical-integrity harm

*Les atteintes à l'intégrité physique des personnes en état de faiblesse et d'ignorance, et plus généralement la commission d'actes criminels ou délictueux sur des individus, majeurs ou mineurs*

**English:** Harm to the physical integrity of people in a state of weakness or ignorance, and more generally the commission of criminal or other unlawful acts against individuals, whether adults or minors

**Nearest entities:**

1. submissiveness
2. crimes of obedience
3. conscientious objection
4. dianetics: the modern science of mental health
5. recovery from cults: help for victims of psychological and spiritual abuse

**Nearest literature expressions:**

1. It is a deviation from freedom of thought, opinion or religion which undermines public order, laws or regulations, or persons' fundamental rights, security or integrity.
2. the goal or effect to create or to exploit the state of mental or physical dependence of people who are participating in its activities
3. use of powers derived from evil spirits, the use of sorcery, or the use of supernatural powers with malicious intent
4. Groups aiming through maneuvers of psychological destabilization to obtain unconditional allegiance from their followers, a diminution in critical thinking, and a rupture with commonly accepted references (ethical, scientific, civic, educational), and entailing dangers for individual freedoms, health, education, and democratic institutions.
5. Moral evil is widely understood to result from ignorance or lack of awareness of the true nature of the Self.
6. a useful though scientifically imprecise concept which refers to an array of complex phenomena resulting in the impairment of the individual's cognitive and social functioning
7. a useful though scientifically imprecise concept which refers to an array of complex phenomena resulting in the impairment of the individual's cognitive and social functioning
8. Religious persecution, meaning violence in which the religion of the persecuted or the persecutor is a factor, affects all religious groups.
9. the criminal activities of sectarian groups that seriously threaten the integrity of the State and citizens
10. CAGs ask ''What do the movements do that is actually or potentially harmful to their own members, to other people and/or to society in general?''

**Nearest interview segments:**

1. Something suspicious, something dangerous, something not normal.
2. I would say it's people participating in something, and then there's a person who has another agenda and kind of manipulates them, and uses, like — their, I don't know — participation, or — I think that's not the right word — but, like, their willingness, and, like, their good [nature] to participate in something.
3. Um — "Missbrauch" [German for "abuse"/"misuse"].
4. Yeah, because when you have a religion, I don't know — you're all having the same thoughts, beliefs, for one thing, and it's like the common knowledge more, it is, okay — whereas with a cult, it's more — um, like, my opinion, it's — I don't know — someone is [misusing, "Missbrauch"] — [?heavily garbled stammering — see notes] — I think that it's the misuse, the distortion, the distortion, the disrespect — like, you're not doing what you should, what you're intending to do, and you're not doing what other people are expecting you to do — from, like, a moral, spiritual point of view, or — yes, ethically, ethically, morally, I think, all these.
5. But I suppose most commonly I'd consider groups of people with some commonly considered 'alternative' beliefs and a person, figure or certain idea as a central element within their communal belief system.
6. something nasty, something really nasty.
7. Yeah, it can be just your boss at work, it can be your military commander, it can be your church commander — your cult commander — but as long as you make people unequal, and somebody's going to be a little afraid to tell you when you touch me on my nipples — you might eventually touch me on my nipples.
8. And then secondly, religious cults.
9. It means committing to rules, committing to maybe a financial engagement, and it comes with giving up some individual freedoms, let's say.
10. Okay, no — that can be — this goes more into the religious and against a particular topic, but in a super extreme way.

### Contesting public order

*La contestation des institutions et les troubles à l'ordre public ; la menace d'atteinte à l'ordre public*

**English:** Contesting institutions and disturbances to public order; the threat of a breach of public order

**Nearest entities:**

1. religion and the social order
2. conscientious objection
3. the constitution of society
4. women and the state
5. demonstration

**Nearest literature expressions:**

1. It is a deviation from freedom of thought, opinion or religion which undermines public order, laws or regulations, or persons' fundamental rights, security or integrity.
2. religious movements may be seen as constituting a threat to political authority
3. NRMs are regarded as a social problem because they pose a challenge to the established social order.
4. religious power and authority are threatened directly or indirectly by the religious claims of these newer groups
5. Sensational Episodes of Violence in NRMs and Public Perceptions of the Growing Threat to Public Order Posed by New and Minority Religions
6. The equation seemed obvious: claims of personal revelation led to violent subversion and unrestrained sexuality, which if unchecked would destroy the social order.
7. Groups aiming through maneuvers of psychological destabilization to obtain unconditional allegiance from their followers, a diminution in critical thinking, and a rupture with commonly accepted references (ethical, scientific, civic, educational), and entailing dangers for individual freedoms, health, education, and democratic institutions.
8. Di Mambro had to face not just internal dissent but also people who knew too much and could have seriously threatened his status.
9. organizing and using evil cults and superstitious sects to undermine law enforcement and public order
10. Conventionalism and conformity (there are conspicuous exceptions, however, in terms of egregiously antinomian groups).

**Nearest interview segments:**

1. Something suspicious, something dangerous, something not normal.
2. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
3. Manipulation, groupe, idéologie.
4. And also something a little bit frightening.
5. And then secondly, religious cults.
6. Yeah, exactly — he's famous for abusing the trust — the trust, the consent, of the people — and using it for his own purposes.
7. I would say it's people participating in something, and then there's a person who has another agenda and kind of manipulates them, and uses, like — their, I don't know — participation, or — I think that's not the right word — but, like, their willingness, and, like, their good [nature] to participate in something.
8. And also, the things that make me address them as such is the fact that they keep trying to include more and more people, and they exert pressure on people to make them join their group — which, I take for granted, comes also with a financial commitment, for example, I don't know.
9. something nasty, something really nasty.
10. These would happen on sight but also kinda behind the doors and there was also several groups with different leaders.

### Legal disputes

*L'importance des démêlés judiciaires*

**English:** The extent of the group's legal entanglements

**Nearest entities:**

1. thelema
2. dramatic denouements
3. dramatic denouement
4. deima
5. deprogramming

**Nearest literature expressions:**

1. it is necessary to prove that there exists a specific danger that an act of indiscriminate mass murder may be committed
2. the hottest [police] file on the planet, the most important of the decade if not of the century
3. the fight against sects should be a necessary aspect of the protection of individual liberties
4. the legal arena is the most significant in terms of its direct impact on the organizational functioning of NRMs
5. The suppression of negative feedback is deemed crucial
6. Compensators come in various kinds and degrees of importance.
7. the leader's part in this system cannot be ignored
8. disengagement/defection
9. the deselection process helped to solidify the commitment of those who stayed
10. a significant legal arsenal permits the State to sanction the sectarian drifts

**Nearest interview segments:**

1. That's interesting.
2. Yes, interesting.
3. He raised this subject specifically — that we should stop pointing fingers at Catholics for being pedophiles and whatever, because we have our own hands kind of dirty too.
4. Something suspicious, something dangerous, something not normal.
5. Please check.
6. Manipulation, groupe, idéologie.
7. And I kind of — yeah, it's interesting.
8. So for you, the main component is this influence thing.
9. Je veux dire les Raëliens, parce qu'on en a parlé à la télé, surtout beaucoup.
10. ça peut être large.

### Financial demands/opacity

*Le caractère exorbitant des exigences financières ; la violation des règlements ou de la loi (travail illégal, formation professionnelle déviante…) et/ou l'opacité de la gestion financière*

**English:** The exorbitant nature of financial demands; violation of regulations or the law (illegal labor, deviant vocational training, etc.) and/or opacity in financial management

**Nearest entities:**

1. conformity
2. regulating religion
3. coercive persuasion
4. misunderstanding cults: searching for objectivity in a controversial field
5. les dérives sectaires

**Nearest literature expressions:**

1. a religion may present barriers to accessibility where its members are of a particular color or caste, conduct their ceremonies in a foreign language, or dress in a way that could be considered culturally inappropriate
2. requiring a high degree of conformity and commitment.
3. Conventionalism and conformity (there are conspicuous exceptions, however, in terms of egregiously antinomian groups).
4. People who lacked a strong sense of boundaries, or conversely, were too rigid and set in their ways, were deemed unsuitable.
5. It is a deviation from freedom of thought, opinion or religion which undermines public order, laws or regulations, or persons' fundamental rights, security or integrity.
6. They frequently refer to the movements as malevolent and destructive cults or totalistic groups that demand hyper-compliance and use coercive power and undue authority.
7. Religions that are apt to need protection are those that the majority may perceive as obnoxious, weird, mysterious, peculiar, and nontraditional.
8. They frequently refer to the movements as malevolent and destructive cults or totalistic groups that demand hypercompliance and use coercive power and undue authority.
9. Evangelism, or concern with proselytization (there may be exceptions here in terms of groups which have isolated themselves in such a way as to inhibit outreach-such groups may be particularly volatile).
10. The characteristics that cause concern can be grouped under the three headings of 'interaction factors,' 'internal factors,' and 'belief factors.'

**Nearest interview segments:**

1. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
2. It means committing to rules, committing to maybe a financial engagement, and it comes with giving up some individual freedoms, let's say.
3. And also, the things that make me address them as such is the fact that they keep trying to include more and more people, and they exert pressure on people to make them join their group — which, I take for granted, comes also with a financial commitment, for example, I don't know.
4. Okay, no — that can be — this goes more into the religious and against a particular topic, but in a super extreme way.
5. Something suspicious, something dangerous, something not normal.
6. And, like, yeah, I kind of just think, "okay, I don't want that information in my brain anymore."  Yeah, so I guess I also see that often — in examples I've seen but don't remember the names of — religion, or some sort of spirituality, was used and then, like, exaggerated.
7. And this is a strictly hierarchical system, so, of course, there's going to be power imbalances, and whenever there's a power imbalance, there's going to be exploitation.
8. And, top — like, not everybody can be there, and, like, kind of a higher status.
9. I would say it's people participating in something, and then there's a person who has another agenda and kind of manipulates them, and uses, like — their, I don't know — participation, or — I think that's not the right word — but, like, their willingness, and, like, their good [nature] to participate in something.
10. C'est pas forcément très bien.

### Infiltration of public authorities

*Les tentatives d'infiltration des pouvoirs publics*

**English:** Attempts to infiltrate public authorities

**Nearest entities:**

1. brainwashing
2. media trickery
3. les dérives sectaires
4. mystical manipulation
5. coercive persuasion

**Nearest literature expressions:**

1. Mystical manipulation .
2. Conspiracy theories about NRMs may involve authorities as well as be motivated from the grassroots.
3. religious movements may be seen as constituting a threat to political authority
4. brainwashing
5. brainwashing
6. brainwashing
7. brainwashing
8. brainwashing
9. brainwashing
10. brainwashing

**Nearest interview segments:**

1. Manipulation, groupe, idéologie.
2. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
3. Somehow they try to convince people to join the cult.
4. These would happen on sight but also kinda behind the doors and there was also several groups with different leaders.
5. So, yeah, kind of those stories of people trying to escape those cults.
6. Des groupements...
7. Et donc je suppose que les instances politiques font en sorte de fermer les sectes, quoi ?
8. This comes to my mind.
9. exploitation?
10. I would say it's people participating in something, and then there's a person who has another agenda and kind of manipulates them, and uses, like — their, I don't know — participation, or — I think that's not the right word — but, like, their willingness, and, like, their good [nature] to participate in something.

### Dubious/exclusive care

*L'offre de soins et de médicaments douteux et exclusive du recours à des pratiques conventionnelles*

**English:** The offer of dubious care and medication, to the exclusion of recourse to conventional practices

**Nearest entities:**

1. bounded choice
2. christian deviations
3. les dérives sectaires
4. a social psychological critique of 'brainwashing' claims about recruitment to new religions
5. mystical manipulation

**Nearest literature expressions:**

1. unusual practices
2. the generalization that cults are detrimental to one's mental health is 'doubtful.'
3. a useful though scientifically imprecise concept which refers to an array of complex phenomena resulting in the impairment of the individual's cognitive and social functioning
4. a useful though scientifically imprecise concept which refers to an array of complex phenomena resulting in the impairment of the individual's cognitive and social functioning
5. the sum of unorthodox and deviant belief systems together with their practices, institutions and personnel
6. NRMs are under suspicion if their members do not avail themselves of publicly available medical services or personnel.
7. methodological doubting
8. Religions that are apt to need protection are those that the majority may perceive as obnoxious, weird, mysterious, peculiar, and nontraditional.
9. a term that refers to controversial groups that are odd and even dangerous to group members and others
10. Conventionalism and conformity (there are conspicuous exceptions, however, in terms of egregiously antinomian groups).

**Nearest interview segments:**

1. Something suspicious, something dangerous, something not normal.
2. des groupes fermés qui endoctrinent des gens en échange d'argent, la plupart du temps, et qui peuvent conduire à des dérives dangereuses.
3. But, yeah, it's a system of belief that spreads and exploits the gullible.
4. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
5. And there's also this weird story about him, because the cult didn't allow access to doctors and medicine, and one of his younger brothers actually died when he was young.
6. And, yeah — say, like, cults — they're just like a really, really niche, small group that's against a topic.
7. Yeah, I guess it's culty, because it's very — it's very niche.
8. But non-conventional religion, so that's why they are, most of the time, concentrated in small groups.
9. The way I've been shown it, like a stereotype — it's because it's like a privileged, small group of people — like, how do you call this — the one where there's only one dollar bill, you know, like this Illuminati thing, you know.
10. But I suppose most commonly I'd consider groups of people with some commonly considered 'alternative' beliefs and a person, figure or certain idea as a central element within their communal belief system.

### Alarming dietary change

*Le changement inquiétant des habitudes alimentaires*

**English:** Alarming changes in dietary habits

**Nearest entities:**

1. concerned relatives
2. self transformation
3. concerned christians
4. food for life
5. exploring new religions

**Nearest literature expressions:**

1. sudden personality change
2. unusual practices
3. a habit of mind… which it has cost much pains and labour to remove
4. the increasing concern, worldwide, with global pollution and environmental damage is especially dominant amongst young people
5. strange
6. the reality of the transformation they experience is both far more immediate and far more complex
7. weird
8. the report shifts in a markedly critical direction concerning 'cults' and new religions
9. The role of women in society is undergoing profound change
10. Cults in our midst: The hidden menace in our everyday lives

**Nearest interview segments:**

1. I think it changes depending on what I've been doing in the last days.
2. Something suspicious, something dangerous, something not normal.
3. This comes to my mind.
4. In these times, people in a big city are growing tomatoes, and they're changing varieties, and they're measuring the sugar [content] of the tomatoes, and they're kind of scientifically looking at growing the tomatoes, and the types, and — yeah.
5. And also something a little bit frightening.
6. C'est pas forcément très bien.
7. Because they are an enclosed group with one strange thing in the full center of their attention (for each of the new age groups)
8. That's interesting.
9. Qui suit des règles.
10. Yeah, but I would be shocked to discover that — whenever there's a power imbalance, people start touching each other in inappropriate places.

### Violation of Republic principles

*La violation des principes fondateurs de la République*

**English:** The violation of the founding principles of the Republic

**Nearest entities:**

1. the constitution of society
2. christian fundamentalism
3. establishment clause
4. constitution
5. christian deviations

**Nearest literature expressions:**

1. It is a deviation from freedom of thought, opinion or religion which undermines public order, laws or regulations, or persons' fundamental rights, security or integrity.
2. the official doctrine applies a fundamentalist reading of the founding texts
3. the criminal activities of sectarian groups that seriously threaten the integrity of the State and citizens
4. sexual transgression
5. freedom of belief
6. the fraudulent abuse of the state of ignorance or weakness
7. Basic Mistrust
8. the religious experience under consideration is not genuine
9. the Commission for Violations of Psychiatry against Human Rights
10. CANCELLED BECAUSE OF COPYRIGHT INFRINGEMENT

**Nearest interview segments:**

1. Yeah, exactly — he's famous for abusing the trust — the trust, the consent, of the people — and using it for his own purposes.
2. And then secondly, religious cults.
3. So, if I understand well, the main feature for you is this — opposing something?
4. Something suspicious, something dangerous, something not normal.
5. attempting to deviate from academic norms through “breaking boundaries”, central figure with authoritative behavior and high controlled dynamic with attempts at isolating members from larger communities
6. This comes to my mind.
7. But, yeah, it's a system of belief that spreads and exploits the gullible.
8. but I might be wrong.
9. Un groupe qui suit une doctrine.
10. exploitation?
