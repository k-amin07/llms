import requests
import time
import math
from enum import Enum

subreddit_rules = {}

last_fetched = time.time()


class REMOVAL_REASON(Enum):
    USER_REMOVED = 1
    MOD_REMOVED = 2


def get_headers():
    """Returns the user agent headers"""
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36"
    }


def check_if_mod_removed(subreddit_name_prefixed, post_id, comment_id):
    global last_fetched
    current_time = time.time()
    time_diff = current_time - last_fetched
    if time_diff < 2:
        time.sleep(math.ceil(2 - time_diff))
    # https://www.reddit.com/svc/shreddit/comments/r/pokemontcg/t3_1i9yyrw/t1_m97ijlu
    url = f"https://www.reddit.com/{subreddit_name_prefixed}/{post_id}/{comment_id}"
    response = requests.get({url}, headers=get_headers())
    last_fetched = time.time()
    resp = response.text
    if "Comment removed by moderator" in resp:
        return REMOVAL_REASON.MOD_REMOVED
    # if("Comment deleted by user" in resp):
    else:
        return REMOVAL_REASON.USER_REMOVED


def get_page(url="https://www.reddit.com/", params=""):
    """Fetch JSON data from a given URL with proper headers."""
    global last_fetched
    current_time = time.time()
    time_diff = current_time - last_fetched
    if time_diff < 2:
        time.sleep(math.ceil(2 - time_diff))
    response = requests.get(f"{url}.json{params}", headers=get_headers())
    last_fetched = time.time()
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data from {url} (Status Code: {response.status_code})")
        return None


def populate_subreddit_rules(sub):
    """Fetch and store subreddit rules."""
    rules_json = get_page(f"https://www.reddit.com/{sub}/about/rules")
    if rules_json is None:
        return

    rules = rules_json.get("rules", [])
    rules_str = "\n".join(
        f"Rule {index + 1}: {rule['description']}" for index, rule in enumerate(rules)
    )
    subreddit_rules[sub] = rules_str
    print(f"Rules for {sub}:\n{subreddit_rules[sub]}\n")


def traverse_comments(data, post_url, depth=2, parent_id=None, parent_link_id=None):
    """Recursively traverse comments."""
    comments_dict = {}

    for comment in data:
        kind = comment["kind"]
        comment_data = comment["data"]
        if depth == 2:
            parent_link_id = comment_data.get("parent_id")

        if kind == "t1":  # Regular comment
            (
                comment_id,
                comment_body,
                replies_data,
                comment_author,
                link_id,
                subreddit,
            ) = map(
                comment_data.get,
                ["id", "body", "replies", "author", "name", "subreddit_name_prefixed"],
            )
            was_removed = comment_data.get("removed_by_category") is not None
            removal_reason = comment_data.get("removed_by_category") or ""

            # Initialize the comment structure
            comments_dict[comment_id] = {
                "body": comment_body or "[removed]",
                "author": comment_author or "[deleted]",
                "parent_id": parent_id,
                "was_removed": was_removed,
                "removal_reason": removal_reason,
                "parent_link_id": parent_link_id,
                "removed_by_moderator": False,
                "link_id": link_id,
                "replies": {},
            }

            if comments_dict[comment_id]["body"] == "[removed]":
                comments_dict[comment_id]["was_removed"] = True
                comments_dict[comment_id]["removed_by_moderator"] = (
                    check_if_mod_removed(subreddit, parent_link_id, link_id)
                    == REMOVAL_REASON.MOD_REMOVED
                )

            # Process nested replies recursively
            if replies_data and isinstance(replies_data, dict) and depth >= 0:
                replies = replies_data.get("data", {}).get("children", [])
                if replies:
                    comments_dict[comment_id]["replies"].update(
                        traverse_comments(
                            replies,
                            post_url,
                            depth=depth - 1,
                            parent_id=comment_id,
                            parent_link_id=parent_link_id,
                        )
                    )

        elif kind == "more" and depth >= 0:  # "More comments" object
            # Process "more" comments by fetching each comment individually
            comment_ids = comment_data.get("children", [])
            for more_id in comment_ids:
                more_url = f"{post_url}/{more_id}"
                more_data = get_page(more_url)
                if more_data and len(more_data) > 1 and depth > 0:
                    # The second item in the JSON is the comments structure
                    additional_comments = more_data[1]["data"]["children"]
                    more_replies = traverse_comments(
                        additional_comments,
                        post_url,
                        depth=depth - 1,
                        parent_id=parent_id,
                        parent_link_id=parent_link_id,
                    )
                    comments_dict.update(more_replies)

    return comments_dict


# def traverse_comments(data):
#     if isinstance(data, list):
#         for comment in data:
#             comment_data = comment["data"]
#             comment_id, comment_body, replies_data, comment_author = list(
#                 map(comment_data.get, ["id", "body", "replies", "author"])
#             )
#     else:
#         pass


# Get 100 posts from reddit front page - default sorting.
limit = 1
r_all_json = get_page(params=f"?limit={limit}")
if not r_all_json:
    exit()

all_posts = r_all_json["data"]["children"]
top_posts_data = {}


for post in all_posts:
    post_data = post["data"]
    post_id, title, sub, content, link = map(
        post_data.get,
        ["id", "title", "subreddit_name_prefixed", "content", "permalink"],
    )

    if sub not in top_posts_data:
        top_posts_data[sub] = {}
        # Get subreddit rules
        populate_subreddit_rules(sub)

    if post_id in top_posts_data[sub]:
        continue

    link = link.rstrip("/")
    print(f"Fetching post: {sub} - {title}")

    top_posts_data[sub][post_id] = {
        "title": title,
        "content": content,
        "link": link,
        "comments": {},
        "remaining_comment_ids": [],
    }

    # Fetch the actual post data
    post_url = f"https://www.reddit.com{link}.json"
    post_page_json = get_page(post_url)

    if not post_page_json or len(post_page_json) < 2:
        continue

    post_comments = post_page_json[1]["data"]["children"]
    comments = traverse_comments(post_comments, post_url)
    remaining_ids = comments.pop("remaining_comment_ids", [])
    top_posts_data[sub][post_id]["comments"] = comments
    top_posts_data[sub][post_id]["remaining_comment_ids"] = remaining_ids
    # Top level comments on the post
    # for comment in post_comments:
    #     # We get 35 comments per post, 35th comment contains ids for the remaining comments.
    #     # For now, we will only deal with the comments that are fetched here but save the ids available in the last comment
    #     if comment["kind"] == "t1":
    #         # Ignore automod
    #         if comment["data"]["author"] == "AutoModerator":
    #             continue
    #         comment_data = comment["data"]
    #         comment_id, comment_body, replies_data = list(
    #             map(comment_data.get, ["id", "body", "replies"])
    #         )
    #         if len(replies_data):
    #             replies = replies_data["data"]["children"]
    #             for reply in replies:
    #                 child_id, child_body,

    #     elif comment["kind"] == "more":
    #         top_posts_data[sub][post_id]["remaining_comment_ids"] = comment["data"][
    #             "children"
    #         ]

    #     pass


# Note: for comments that are removed, some might be removed by moderators but we may not get the mod removal reason - or the fact that they were removed by a moderatory
# Once the data is fetched, during refetching, build a url like this for the removed comments that were not previously removed.
# https://www.reddit.com/svc/shreddit/comments/r/pokemontcg/t3_1i9yyrw/t1_m97ijlu
# The second id is the available against the "name" key. the first one can be extracted from all top level comments using parent_id or link_id

# This is a preexisting dataset that I can actually use - but it has over 14B rows.
# https://clickhouse.com/docs/en/getting-started/example-datasets/reddit-comments
# Pick a row, extract subreddit, and link_id. Go to reddit.com/r/<subreddit>/comments/<link_id>
