#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce


#--------------------------------------------------------------------------------------------------------------------
#	distinct([numTasks]))	Return a new dataset that contains the distinct elements of the source dataset..
#


def distinct_master(data1,data2):
    from pycompss.api.api import compss_wait_on

    if (len(data1) > len(data2)):
        data_return = [[] for i in xrange(0,len(data1)) ]
        for i in xrange(0,len(data2)):
            data_return[i] = distinct(data1[i],data2[i]) 

        for y in xrange(len(data2),len(data1)):
            data_return[y] = data1[y] 

    elif (len(data1) < len(data2)):
        data_return = [[] for i in xrange(0,len(data2)) ]
        for i in xrange(0,len(data1)):
           data_return[i] = distinct(data1[i],data2[i]) 

        for y in xrange(len(data1),len(data2)):
             data_return[y] = data2[y]
          
    else:
        data_return = [[] for i in xrange(0,len(data1)) ]
        for i in xrange(0,len(data1)):
           data_return[i] = distinct(data1[i],data2[i])
        
    data_result = mergeReduce(distinct, data_return)
        
    data_result = compss_wait_on(data_result)

    return data_result



@task(returns=list)
def distinct(list1,list2):
    return [x for x in list1 if x not in list2]+[x for x in list2 if x not in  list1]



#--------------------------------------------------------------------------------------------------------------------
#	union(otherDataset)	Return a new dataset that contains the union of the elements in the source dataset and the argument.
#
def union_master(data1,data2):
    from pycompss.api.api import compss_wait_on

    
    if (len(data1) > len(data2)):
        data_return = [[] for i in xrange(0,len(data1)) ]
        for i in xrange(0,len(data2)):
            data_return[i] = union(data1[i],data2[i]) 

        for y in xrange(len(data2),len(data1)):
            data_return[y] = data1[y] 

    elif (len(data1) < len(data2)):
        data_return = [[] for i in xrange(0,len(data2)) ]
        for i in xrange(0,len(data1)):
           data_return[i] = union(data1[i],data2[i]) 

        for y in xrange(len(data1),len(data2)):
             data_return[y] = data2[y]
          
    else:
        data_return = [[] for i in xrange(0,len(data1)) ]
        for i in xrange(0,len(data1)):
           data_return[i] = union(data1[i],data2[i])
        
    data_return = compss_wait_on(data_return)

    return data_return


@task(returns=list)
def union(list1,list2):
    return list(set( list1 + list2 ))



#--------------------------------------------------------------------------------------------------------------------
#	intersection(otherDataset)	Return a new RDD that contains the intersection of elements in the source dataset and the argument.
#

def intersection_master(data1,data2):
    from pycompss.api.api import compss_wait_on

    data_result =[]
    if len(data1)>=len(data2):
        for i in xrange(0,len(data1)):
            for j in xrange(0,len(data2)):
                data_return = intersection(data1[i],data2[j])  
                data_result = union(data_return,data_result)
    else:
        for i in xrange(0,len(data2)):
            for j in xrange(0,len(data1)):
                data_return = intersection(data1[i],data2[j])  
                data_result = union(data_return,data_result)
        
    data_result = compss_wait_on(data_result)

    return data_result



@task(returns=list)
def intersection(list1,list2):
    return  [x for x in list1 if x in list2]
  


#--------------------------------------------------------------------------------------------------------------------
#	Reduce
#
#@task(dic1=INOUT)
#def reduce(dic1,dic2):
#    for k in dic2:
#        if k in dic1:
#            dic1[k] += dic2[k]
#        else:
#            dic1[k] = dic2[k]



#
#   Split the data in n parts
#
def chunks(data, n, balanced=True):
    if not balanced or not len(data) % n:
        for i in xrange(0, len(data), n):
            yield data[i:i + n]
    else:
        rest = len(data) % n
        start = 0
        while rest:
            yield data[start: start+n+1]
            rest -= 1
            start += n+1
        for i in xrange(start, len(data), n):
            yield data[i:i + n]



if __name__ == "__main__":



    numFrag = 2
    data1 = [100,200,300,2,400,9,10,11,-2,-9,-8]
    data2 = [400,500,600,99,2,1,-8]
    data_result = []

    #
    # Create the chunks
    #
    data1 = [d for d in chunks(data1, len(data1)/numFrag)]
    data2 = [d for d in chunks(data2, len(data2)/numFrag)]

    print data1
    print data2

    
    # Union
    # Intersection
    # Distinct


    #data_result = union_master(data1,data2)
    data_result = distinct_master(data1,data2)
    #data_result = intersection_master(data1,data2)
    

    print "Result: " + str(data_result)