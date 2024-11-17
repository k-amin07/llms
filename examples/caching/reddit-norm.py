import csv
import os
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from redisvl.extensions.llmcache import SemanticCache
from sklearn.metrics import cohen_kappa_score


class RedditCache:
    def __init__(self, dataset_path="", redis_url="redis://localhost:6379", similarity_threshold=0.7) -> None:
        self.redis_url = redis_url
        self.similarity_threshold = similarity_threshold
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(current_dir, dataset_path)
        self.TP, self.FP, self.TN, self.FN, self.rule_match = 0,0,0,0,0
        self.results = []
        self.cache_rating = []
        self.llm_rating = []
        self.results_path = self.data_path[:-4] + "-results.csv"
        self.cm_path = self.data_path[:-4] + "-cm.csv"
    
    def load_data(self):
        with open(self.data_path, 'r') as dataset:
            self.data = dataset.readlines()
        # self.data = ["I completely agree with your stance cunt, open source developers should get compensated well","I completely agree with your stance, open source developers should get compensated well","what a dumb cunt lol what a pussy","nobody cares you dumb cunt","wow kinda dumb u and your cunt daughter","no one gives a fuck what this batshit community thinks","savage this pussy is a bumbling idiot lol","nobody cares you pussy faget","you re a dumb cunt lol","you are one dumb cunt","u leave nobody wants u here cuckface","no one cares cunt","wow you re an idiot grow up","nobody cares cunt","nobody cares douchebag","and flo is equally shit nobody gives a fuck about you flo you dumb cunt","absolute dumb cunt","i bet you re a vegan cunt","no one wants to spoon with your dumbass community college retard chink headass bitch rawdog naved","lol nobody cares fuck off","then fuck off lol nobody cares","fuck u dumb tards die in hell"]
    
    def init_cache(self, name="redditnorm",prefix="redditnorm",distance_threshold=0.25, use_sim_threshold=False)->None:
        '''Initializes the Redis Semantic Cache
            Params:
                name: Name for the redis collection. Defaults to llmcache
                prefix: prefix for keys in llm cache. Defaults to 
                distance_threshold: custom threshold used if use_sim_threshold is set to False.
                use_sim_threshold: Whether to use the same similarity threshold as the caching class
        '''
        
        self.distance_threshold = distance_threshold

        self.llmcache = SemanticCache(
            name=name,
            prefix=prefix,
            redis_url=self.redis_url,
            distance_threshold=1-self.similarity_threshold if use_sim_threshold else distance_threshold
        )

    def init_llm(self, model_name="gpt-4o-mini", base_url=None):
        self.llm = ChatOpenAI(model_name=model_name, base_url=base_url)
        assistant_system_message = """
        You are a content moderator on Reddit. Here is the Reddiquette, i.e., list of rules that all content on Reddit must adhere to. 
        All posts and comments must:

            1. Remember the human. When you communicate online, all you see is a computer screen. When talking to someone you might want to ask yourself "Would I say it to the person's face?" or "Would I get jumped if I said this to a buddy?"

            2. Adhere to the same standards of behavior online that you follow in real life.

            3. Read the rules of a community before making a submission. These are usually found in the sidebar.

            4. Read the reddiquette. Read it again every once in a while. Reddiquette is a living, breathing, working document which may change over time as the community faces new problems in its growth.

            5. Moderate based on quality, not opinion. Well written and interesting content can be worthwhile, even if you disagree with it.

            6. Use proper grammar and spelling. Intelligent discourse requires a standard system of communication. Be open for gentle corrections.

            7. Keep your submission titles factual and opinion free. If it is an outrageous topic, share your crazy outrage in the comment section.

            8. Look for the original source of content, and submit that. Often, a blog will reference another blog, which references another, and so on with everyone displaying ads along the way. Dig through those references and submit a link to the creator, who actually deserves the traffic.

            9. Post to the most appropriate community possible. Also, consider cross posting if the contents fits more communities.

            10. Vote. If you think something contributes to conversation, upvote it. If you think it doesn't contribute to the community it's posted in or is off-topic in a particular community, downvote it.

            11. Search for duplicates before posting. Redundancy posts add nothing new to previous conversations. That said, sometimes bad timing, a bad title, or just plain bad luck can cause an interesting story to fail to get noticed. Feel free to post something again if you feel that the earlier posting didn't get the attention it deserved and you think you can do better.

            12. Link to the direct version of a media file if the page it was found on isn't the creator's and doesn't add additional information or context.

            13. Link to canonical and persistent URLs where possible, not temporary pages that might disappear. In particular, use the "permalink" for blog entries, not the blog's index page.

            14. Consider posting constructive criticism / an explanation when you downvote something, and do so carefully and tactfully.

            15. Actually read an article before you vote on it (as opposed to just basing your vote on the title).

            16. Feel free to post links to your own content (within reason). But if that's all you ever post, or it always seems to get voted down, take a good hard look in the mirror — you just might be a spammer. A widely used rule of thumb is the 9:1 ratio, i.e. only 1 out of every 10 of your submissions should be your own content.

            17. Posts containing explicit material such as nudity, horrible injury etc, add NSFW (Not Safe For Work) and tag. However, if something IS safe for work, but has a risqué title, tag as SFW (Safe for Work). Additionally, use your best judgement when adding these tags, in order for everything to go swimmingly.

            18. State your reason for any editing of posts. Edited submissions are marked by an asterisk (*) at the end of the timestamp after three minutes. For example: a simple "Edit: spelling" will help explain. This avoids confusion when a post is edited after a conversation breaks off from it. If you have another thing to add to your original comment, say "Edit: And I also think..." or something along those lines.

            19. Use an "Innocent until proven guilty" mentality. Unless there is obvious proof that a submission is fake, or is whoring karma, please don't say it is. It ruins the experience for not only you, but the millions of people that browse Reddit every day.

            20. Read over your submission for mistakes before submitting, especially the title of the submission. Comments and the content of self posts can be edited after being submitted, however, the title of a post can't be. Make sure the facts you provide are accurate to avoid any confusion down the line.

        All posts and comments must not:

            21. Engage in illegal activity.

            22. Post someone's personal information, or post links to personal information. This includes links to public Facebook pages and screenshots of Facebook pages with the names still legible. We all get outraged by the ignorant things people say and do online, but witch hunts and vigilantism hurt innocent people too often, and such posts or comments will be removed. Users posting personal info are subject to an immediate account deletion. If you see a user posting personal info, please contact the admins. Additionally, on pages such as Facebook, where personal information is often displayed, please mask the personal information and personal photographs using a blur function, erase function, or simply block it out with color. When personal information is relevant to the post (i.e. comment wars) please use color blocking for the personal information to indicate whose comment is whose.

            23. Repost deleted/removed information. Remember that comment someone just deleted because it had personal information in it or was a picture of gore? Resist the urge to repost it. It doesn't matter what the content was. If it was deleted/removed, it should stay deleted/removed.

            24. Be (intentionally) rude at all. By choosing not to be rude, you increase the overall civility of the community and make it better for all of us.

            25. Follow those who are rabble rousing against another redditor without first investigating both sides of the issue that's being presented. Those who are inciting this type of action often have malicious reasons behind their actions and are, more often than not, a troll. Remember, every time a redditor who's contributed large amounts of effort into assisting the growth of community as a whole is driven away, projects that would benefit the whole easily flounder.

            26. Ask people to Troll others on Reddit, in real life, or on other blogs/sites. We aren't your personal army.

            27. Conduct personal attacks on other commenters. Ad hominem and other distracting attacks do not add anything to the conversation.

            28. Start a flame war. Just report and "walk away". If you really feel you have to confront them, leave a polite message with a quote or link to the rules, and no more.

            29. Insult others. Insults do not contribute to a rational discussion. Constructive Criticism, however, is appropriate and encouraged.

            30. Troll. Trolling does not contribute to the conversation.

            31. Take moderation positions in a community where your profession, employment, or biases could pose a direct conflict of interest to the neutral and user driven nature of Reddit

            32. Make comments that lack content. Phrases such as "this", "lol", and "I came here to say this" are not witty, original, or funny, and do not add anything to the discussion.

            33. Announce your vote (with rare exceptions). "Upvote" and "Downvote" aren't terribly interesting comments and only increase the noise to signal ratio.

            34. Complain about other users reposting/rehosting stories, images, videos, or any other content. Users should give credit where credit should be given, but if someone fails to do so, and is not causing harm, please either don't point it out, or point it out politely and leave it at that. They are only earning karma, which has little to no use at all.

            35. Complain about the votes you do or do not receive, especially by making a submission voicing your complaint. You may have just gotten unlucky. Try submitting later or seek out other communities to submit to. Millions of people use Reddit; every story and comment gets at least a few up/downvotes. Some up/downvotes are by Reddit to fuzz the votes in order to confuse spammers and cheaters. This also includes messaging moderators or admins complaining about the votes you did or did not receive, except when you suspect you've been targeted by vote cheating by being massively up/downvoted.

            36. Complain about reposts. Just because you have seen it before doesn't mean everyone has. Votes indicate the popularity of a post, so just vote. Keep in mind that linking to previous posts is not automatically a complaint; it is information.

            37. Complain about cross posts. Just because you saw it in one place, doesn't mean everyone has seen it. Just vote and move on.
        
        Given a comment, your task is to determine whether the comment goes against these rules. If a comment breaks a rule, return "Remove - Rule: " followed by the rule number that the comment breaks from the list of rules mentioned above, otherwise return "KEEP". Please adhere only to the provided rules. Avoid using any tools like {tools},{tool_names} for this task and do not apply special formatting to response
        """
        noop_tool = Tool(
            name="NoOp",
            func=lambda x: "No tools used",
            description="This is a placeholder tool and should not be used."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", assistant_system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),  # The user message will be inserted here
                ("assistant", "{agent_scratchpad}")  # Placeholder for the assistant’s thoughts
            ]
        )

        self.agent = create_react_agent(
            llm=self.llm,
            prompt=prompt,
            tools=[noop_tool],
        )
        self.agent_executor = AgentExecutor(agent=self.agent,verbose=True, tools=[noop_tool])
        set_llm_cache(InMemoryCache())
    
    def extract_response(self, response):
        if response == "KEEP":
            decision = response
            rule = ""
        else:
            decision = "REMOVE"
            rule = "Rule: " + response.split("Rule: ")[1]
        return decision, rule

    
    def process_data(self, limit = 20, process_all = False):
        if limit > len(self.data) or process_all:
            limit = len(self.data)
        
        for idx, comment in enumerate(self.data):
            if(idx == limit):
                break

            print("Processing comment: {}".format(comment))
            chat_history = []
            was_cached = False
            cache_decision, cache_rule = "",""
            if resp := self.llmcache.check(prompt=comment):
                cache_response = resp[0]["response"]
                cache_decision, cache_rule = self.extract_response(cache_response)
                was_cached = True
                
            try:
                response = self.agent_executor.invoke({"input":comment,"chat_history":chat_history})
                print(response)
                response = response['Output']
            except Exception as e:
                response = str(e).split("`")[-2]
            decision, rule = self.extract_response(response)
            tp, tn, fp, fn, rule_match = 0,0,0,0,0

            # rating criteria: REMOVE = 0, KEEP = 1
            
            if was_cached:
                if decision == "REMOVE" and cache_decision == "REMOVE":
                    self.TP += 1
                    tp = 1
                    self.cache_rating.append(0)
                    self.llm_rating.append(0)
                elif decision == "KEEP" and cache_decision == "KEEP":
                    self.TN += 1
                    tn = 1
                    self.cache_rating.append(1)
                    self.llm_rating.append(1)
                elif decision == "KEEP" and cache_decision == "REMOVE":
                    self.FN += 1
                    fn = 1
                    self.cache_rating.append(0)
                    self.llm_rating.append(1)
                elif decision == "REMOVE" and cache_decision == "KEEP":
                    self.FP += 1
                    fp = 1
                    self.cache_rating.append(1)
                    self.llm_rating.append(0)
            if cache_rule == rule and decision == "REMOVE":
                self.rule_match += 1
                rule_match = 1
            

            chat_history.append(HumanMessage(comment))
            chat_history.append(AIMessage(response))
            self.llmcache.store(
                prompt=comment,
                response=response
            )
            
            self.results.append([comment, response, rule, decision, cache_decision, was_cached, tp, tn, fp, fn, rule_match])
    
    def process_results(self):
        kappa = cohen_kappa_score(self.llm_rating, self.cache_rating)
        if np.isnan(kappa):
            if np.array_equal(self.llm_rating, self.cache_rating):
                kappa = 1.0
                
        prec_denominator = self.TP + self.FP
        precision = (self.TP / prec_denominator) if prec_denominator > 0 else 0
        precision = round(precision,2)
        rec_denominator = self.TP + self.FN
        recall = (self.TP / rec_denominator) if rec_denominator > 0 else 1
        recall = round(recall,2)
        print({"precision":precision})
        print({"recall":recall})
        print({"kappa":kappa})

        headers = ["comment", "response", "rule", "decision","cache_decision", "was_cached", "TP", "TN", "FP", "FN", "Rule Match"]

        with open(self.results_path, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(self.results)

        headers = ["precision", "recall", "kappa"]
        with open(self.cm_path, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerow([precision,recall,kappa])

    
    def close(self):
        '''
        Clears the redis cache for next experiment
        '''
        self.llmcache.delete()


rc = RedditCache(dataset_path="../../datasets/reddit-norm-violations/macro-norm-violations-n10-t0-misogynistic-slurs.csv")
rc.init_cache()
rc.init_llm()
rc.load_data()
rc.process_data(process_all=True)
rc.process_results()