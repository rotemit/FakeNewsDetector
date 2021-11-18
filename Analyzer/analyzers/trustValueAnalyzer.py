from modules.Account import Account
from modules.Group import Group
from modules.Page import Page

# UTV = user trust level
# aua = age of user account, fd = friendship duration, tf = total friends, mf = mutual friends
# all computations are according to Nadav's article

def analyze_facebook(obj):
    if isinstance(obj, Account):
        return analyze_account(obj)
    if isinstance(obj, Group):
        return analyze_group(obj)
    if isinstance(obj, Page):
        return analyze_page(obj)
    return -1

def analyze_account(account):
    aua = account.age
    fd = account.friendship_duration
    tf = account.total_friends
    mf = account.mutual_friends

    if fd == 0:
        return -1
    # if aua == 0 or fd == 0 or tf ==0 or mf == 0:
    #     return -1
        
    # Thresholds
    T_aua = 244.34 #days
    T_fd = 17.12 #days
    T_tf = 23.82 #people
    T_mf = 37 #people

    # User credibility attributes
    U_aua = 1 if aua >= T_aua else aua/T_aua
    U_tf = 1 if tf >= T_tf else tf/T_tf
    userCredibility = (U_aua + U_tf)/2

    # Connection strength attributes
    C_fd = 1 if fd >= T_fd else fd/T_fd
    C_mf = 1 if mf >= T_mf else mf/T_mf
    connectionStrength = (C_fd + C_mf)/2

    # UTV = (U*|U| + C*|C|) / |U + C|
    UTV = (userCredibility*2 + connectionStrength*2)/4
    return UTV


def analyze_group(group):
    ag = group.age
    priv = group.isPrivate
    vis = group.isVisible
    tf = group.friends
    mf = group.mutual_friends

    # if ag == 0 or tf ==0 or mf == 0:
    #     return -1

    # Thresholds
    T_ag = 244.34  # days
    T_tf = 23.82  # people
    T_mf = 37  # peoplepyh
    T_priv = 1
    T_vis = 1

    # group credibility attributes
    U_ag = 1 if ag >= T_ag else ag / T_ag
    U_tf = 1 if tf >= T_tf else tf / T_tf

    U_priv = 1 if priv >= T_priv else priv / T_priv
    U_vis = 1 if vis >= T_vis else vis / T_vis
    U_priv_vis = (U_priv + U_vis)/2
    userCredibility = (U_ag + U_tf + U_priv_vis) / 3

    # Connection strength attributes
    C_mf = 1 if mf >= T_mf else mf / T_mf
    connectionStrength = C_mf

    # UTV = (U*|U| + C*|C|) / |U + C|
    GTV = (userCredibility * 3 + connectionStrength * 1) / 4
    return GTV


def analyze_page(page):
    ap = page.age
    fol = page.followers
    lik = page.num_of_likes
    mf = page.mutual_friends
    tf = max(fol, lik)

    # if ap == 0 or tf == 0 or mf == 0:
    #     return -1

    # Thresholds
    T_ap = 244.34  # days
    T_tf = 23.82  # people
    T_mf = 37  # people

    # Page credibility attributes
    U_ap = 1 if ap >= T_ap else ap / T_ap
    U_tf = 1 if tf >= T_tf else tf / T_tf

    userCredibility = (U_ap + U_tf ) / 2

    # Connection strength attributes
    C_mf = 1 if mf >= T_mf else mf / T_mf
    connectionStrength = C_mf

    # UTV = (U*|U| + C*|C|) / |U + C|
    PTV = (userCredibility * 2 + connectionStrength * 1) / 3
    return PTV