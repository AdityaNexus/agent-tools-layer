a = [1,4,3,3,2]

def find(a):
    ass = 1
    dss = 1
    ans  = 0
    for i in range(len(a)-1):
        if a[i] < a[i+1]:
            ass += 1
            dss = 1
        elif a[i] > a[i+1]:
            dss  += 1
            ass = 1
        else:
            ass= 1
            dss = 1
        ans = max(ass,dss,ans)
    return ans

print(find(a))