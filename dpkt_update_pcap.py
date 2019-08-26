#!/usr/bin/env python3
#encoding:utf-8
#Auther:VChao
#2017/04/14
#This script is designed to modify the port of the pcap file.
#To test if whether the wireshark could recognize the traffic.
#-----------------------------------------------------------#
#测试结果:
#本脚本可以对流量内容进行修改
#初衷是想,看看WireShark能不能将其识别出来,结果很令人悲哀
#当然,如果你硬解码式的将其内容解析为,是肯定可以,这也是早就知道的
#-----------------------------------------------------------#
#而且,内容转化之后,内容也发生了变化,时间戳不准确了
#经过测试发现,即使不指定时间戳,也就是直接使用系统当前的时间
#得到的文件也是不准确的
#这说明,这部分的源码是不对的
#-------------#
#结果是因为WireShark那边的时间没有显示清楚,尴尬

import dpkt

def main():
    with open("ssh.pcap","r") as fin:
        with open("res.pcap","w") as fout:
            pcapin = dpkt.pcap.Reader(fin)
            pcapout = dpkt.pcap.Writer(fout,nano= True)
            for ts,buf in pcapin:
                Eth = dpkt.ethernet.Ethernet(buf)
                #下面是想能重组这个数据包
                ip = Eth.data
                tcp = ip.data
            
                if tcp.dport == 22:
                    tcp.dport = 5022
                if tcp.sport == 22:
                    tcp.sport = 5022

                ip.data = tcp
                temp = dpkt.ethernet.Ethernet(src = Eth.src,dst = Eth.dst,type= Eth.type,data = ip)
                #下面这句很关键,必须将其转化为字符串格式才可以
                pcapout.writepkt(buf,ts = ts)
            pcapout.close()
        
if __name__ == "__main__":
    main()
