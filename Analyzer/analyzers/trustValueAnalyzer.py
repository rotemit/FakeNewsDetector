from modules.Account import Account
from modules.Group import Group
from modules.Page import Page


def analyze_facebook(obj):
    if isinstance(obj, Account):
        return analyze_account(obj)
    if isinstance(obj, Group):
        return analyze_group(obj)
    if isinstance(obj, Page):
        return analyze_page(obj)
    return -1

# UTV = user trust level
# aua = age of user account, fd = friendship duration, tf = total friends, mf = mutual friends
# all computations are according to Nadav's article
def analyze_account(account):
    aua = account.age
    fd = account.friendship_duration
    tf = account.total_friends
    mf = account.mutual_friends

    # in case they are not friends
    if fd == 0:
        return -1

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
    tf = group.friends
    mf = group.mutual_friends

    # Thresholds
    T_ag = 244.34  # days
    T_tf = 50000  # people
    T_mf = 37  # people

    # group credibility attributes
    U_ag = 1 if ag >= T_ag else ag / T_ag
    U_tf = 1 if tf >= T_tf else tf / T_tf
    groupCredibility = (U_ag + U_tf) / 2

    # Connection strength attributes
    C_mf = 1 if mf >= T_mf else mf / T_mf
    connectionStrength = C_mf

    GTV = (groupCredibility * 2 + connectionStrength * 1) / 3
    return GTV


def analyze_page(page):
    ap = page.age
    fol = page.followers
    lik = page.num_of_likes
    mf = page.mutual_friends
    tf = max(fol, lik)

    # Thresholds
    T_ap = 244.34  # days
    T_tf = 25000  # people
    T_mf = 37  # people

    # Page credibility attributes
    U_ap = 1 if ap >= T_ap else ap / T_ap
    U_tf = 1 if tf >= T_tf else tf / T_tf
    pageCredibility = (U_ap + U_tf ) / 2

    # Connection strength attributes
    C_mf = 1 if mf >= T_mf else mf / T_mf
    connectionStrength = C_mf

    PTV = (pageCredibility * 2 + connectionStrength * 1) / 3
    return PTV