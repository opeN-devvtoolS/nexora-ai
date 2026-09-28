import json


def generate_issue_id(issues):

    return (f"ISS-{000+len(issues)+1:03d}")



def create_issue(issues,product_ID,issue_type,priority,evidence):

    if(issue_exists(issues,product_ID,issue_type)):
        return

    
    issues.append({
        "issue_id": generate_issue_id(issues),
        "product_id": product_ID,
        "issue_type": issue_type,
        "status": "OPEN",
        "priority": priority,
        "evidence": evidence })



def issue_exists(issues, product_id, issue_type):

    for issue in issues:
        if (issue["product_id"] == product_id and issue["issue_type"] == issue_type):
            return True
             
    return False


def save_issues(issues):

    with open("data/issues.json","w") as file:
        json.dump(issues,file,indent=4)


def load_issues():

    try:
        with open("data/issues.json","r") as file:
            return json.load(file)
        
    except FileNotFoundError:
        return []


def find_issue(issues,issue_id):

    for issue in issues:
        if  (issue["issue_id"] == issue_id):
            return issue


    return None


def update_issue(issues,issue_id,new_status):

    issue = find_issue(issues,issue_id)

    if(issue):
        if (new_status == "OPEN" or new_status == "IN_PROGRESS" or new_status == "RESOLVED"):
            issue["status"] = new_status
            return True

    return False


def delete_issue(issues,issue_id):
    issue = find_issue(issues,issue_id)

    if(issue):
        issues.remove(issue)
        return True

    return False