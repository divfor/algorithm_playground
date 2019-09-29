#!/usr/bin/env python3
# Turns a pcap file with http gzip compressed data into plain text, making it
# easier to follow.

import socket 
import dpkt
from binascii import hexlify, unhexlify

def parse_pcap_file(filename):
    f = open(filename, 'rb')
    pcap = dpkt.pcap.Reader(f)
    for ts, buf in pcap:
        eth = dpkt.ethernet.Ethernet(buf)
        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
            continue
        ip = eth.data
        if ip.p != dpkt.ip.IP_PROTO_UDP:
            continue
        udp = ip.data
        srcip = socket.inet_ntoa(ip.src)
        dstip = socket.inet_ntoa(ip.dst)
        print("%s,%s,%s,%s,%s" % (srcip, dstip, udp.sport, udp.dport, hexlify(udp.data)))
    f.close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) <= 1:
        print("%s <pcap filename>" % sys.argv[0])
        sys.exit(2)

    parse_pcap_file(sys.argv[1])
