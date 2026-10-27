GPU Adapter / Computing Task	Global	Local	Total Time	Average Time	Instance Count	SIMD Width	SVM Usage Type	Spill Memory Size	General Register File Size	Host-to-Device	Device-to-Host	Active	Stalled	Idle	XVE Threads Occupancy	Thread Dispatcher Active	ALU0 and ALU1 active	ALU0 and XMX active	Multiple pipes active	ALU0 active	ALU1 active	XMX active	ALU0 Instructions	ALU1 Instructions	XMX instructions	Barrier and Sync Instructions	Instruction Fetch	Barrier	SBID	Dist or Acc	Send	Pipe	Control	Other	Instruction Cache Miss Ratio	Busy	Stalled	SQ Full	Read	Write	Input Available	Output Ready	Miss Ratio	TLB Misses	Read	Write	Busy	Input Available	Output Ready
launch_add_n_f32_strided<(int)9>(void, ggml_backend_sycl_context&, std::array<ggml_tensor const*, (unsigned long)9> const&, ggml_tensor*, unsigned int, float, float)::{lambda(sycl::_V1::nd_item<(int)1>)#1}	10240	256	9.755ms	609.672usec	16	16		0	128	0.0 B 	0.0 B 	0.0%	90.8%	9.2%	45.4%	0.0%	0.0%	0.0%	0.0%	0.0%	0.0%	0.0%	17,280	839,040	0	0	0.0%	0.0%	90.8%	0.0%	0.0%	0.0%	0.0%	0.0%	5.5%	1.3%	0.0%	0.0%	1.113	1.101	0.1%	0.0%	50.6%	304	0.024	1.100	0.0%	0.0%	0.0%
launch_add_n_f32_strided<(int)2>(void, ggml_backend_sycl_context&, std::array<ggml_tensor const*, (unsigned long)9> const&, ggml_tensor*, unsigned int, float, float)::{lambda(sycl::_V1::nd_item<(int)1>)#1}	10240	256	19.689ms	615.284usec	32	16		0	128	0.0 B 	0.0 B 	0.0%	86.2%	13.8%	43.1%	0.0%	0.0%	0.0%	0.0%	0.0%	0.0%	0.0%	768	521,216	0	0	0.0%	0.0%	86.2%	0.0%	0.0%	0.0%	0.0%	0.0%	15.1%	1.4%	0.0%	0.0%	1.039	1.265	0.1%	0.0%	55.4%	601	0.022	1.263	0.0%	0.0%	0.0%



Address	Source Line	Assembly
0x87c0	0	Block 1:
0x87c0	0	(W) mov (16|M0) r127.0<1>:ud 0x0:ud
0x87d0	0	(W) and (1|M0) r127.2<1>:ud r0.0<0;1,0>:ud 0xFFFFFFC0:ud
0x87e0	0	(W) and (1|M0) r127.0<1>:uw r0.4<0;1,0>:uw 0xFF:uw
0x87f0	0	(W) add (1|M0) r127.2<1>:ud r127.2<0;1,0>:ud 0x160:ud {I@2}
0x8800	0	(W) add (1|M0) r127.2<1>:ud r127.2<0;1,0>:ud 0x0:ud {I@1}
0x8810	0	(W) mad (1|M0) r127.0<1>:ud r127.2<0;0>:ud r127.0<0;0>:uw 0x40:uw {I@1}
0x8820	0	(W) mov (8|M0) r2.0<1>:ud r1.0<1;1,0>:ud
0x8830	0	(W) send.ugm (1|M0) r1 r127 null:0 0xFF000000 0x6219D500 {A@1,$0} [, msg-length:1, resp-length:1, header:yes, func-control:1d500]
0x8840	0	(W) and (1|M0) r127.0<1>:ud r0.0<0;1,0>:ud 0xFFFFFFC0:ud {$0.src}
0x8850	0	(W) add (1|M0) r127.0<1>:ud r127.0<0;1,0>:ud 0x0:ud {I@1}
0x8860	0	(W) send.ugm (1|M0) r3 r127 null:0 0xFF000000 0x6249F500 {I@1,$1} [, msg-length:1, resp-length:4, header:yes, func-control:1f500]
0x8870	0	(W) send.ugm (1|M0) r7 r127 null:0 0xFF100000 0x6219D500 {$2} [, msg-length:1, resp-length:1, header:yes, func-control:1d500]
0x8880	0	(W) send.ugm (1|M0) r8 r127 null:0 0xFF140000 0x6219C500 {$3} [, msg-length:1, resp-length:1, header:yes, func-control:1c500]
0x8890	0	(W) mov (16|M0) r123.0<1>:ud r0.0<1;1,0>:ud {Compacted}
0x8898	0	(W) or (1|M0) cr0.0<1>:ud cr0.0<0;1,0>:ud 0x400004C0:ud {A@1}
0x88a8	0	(W) mul (1|M0) acc0.0<1>:ud r2.0<0;1,0>:ud r123.2<0;1,0>:uw {A@1}
0x88b8	0	(W) macl (1|M0) r9.0<1>:ud r2.0<0;1,0>:ud r123.1<0;1,0>:ud {Compacted}
0x88c0	0	(W) mul (1|M0) acc0.0<1>:ud r2.0<0;1,0>:ud r123.2<0;1,0>:uw
0x88d0	0	(W) mach (1|M0) r10.0<1>:d r2.0<0;1,0>:ud r123.1<0;1,0>:ud
0x88e0	0	mov (16|M0) r11.0<4>:uw r1.0<1;1,0>:uw {$0.dst}
0x88f0	0	(W) mov (1|M0) r9.1<1>:d r10.0<0;1,0>:d {I@2}
0x8900	0	add (16|M0) r124.0<1>:q r9.0<0;1,0>:q r11.0<4;1,0>:uw {I@1}
0x8910	0	mov (16|M0) r126.0<1>:d r124.0<2;1,0>:d {Compacted,I@1}
0x8918	0	cmp (16|M0) (lt)f0.0 null<1>:d r126.0<1;1,0>:ud r2.6<0;1,0>:ud {I@1}
0x8928	0	(~f0.0) goto (16|M0) bb_3 bb_3
0x8938	0	Block 2:
0x8938	0	mov (16|M0) r1.0<1>:d r124.0<2;1,0>:d {Compacted}
0x8940	0	(W) mul (16|M0) acc0.0<1>:ud r1.0<1;1,0>:ud r7.20<0;1,0>:uw {@1,$2.dst}
0x8950	0	mach (16|M0) r19.0<1>:d r1.0<1;1,0>:ud r7.10<0;1,0>:ud
0x8960	0	(W) mul (16|M0) acc0.0<1>:d r7.10<0;1,0>:ud r124.2<4;1,0>:uw
0x8970	0	macl (16|M0) r34.0<1>:d r7.10<0;1,0>:ud r124.1<2;1,0>:d
0x8980	0	add (16|M0) acc0.0<1>:d r19.0<1;1,0>:d r34.0<1;1,0>:d {I@1}
0x8990	0	add (16|M0) r38.0<1>:d acc0.0<1;1,0>:d r126.0<1;1,0>:d
0x89a0	0	shr (16|M0) r39.0<1>:ud r38.0<1;1,0>:ud r7.11<0;1,0>:d {I@1}
0x89b0	0	(W) mul (16|M0) acc0.0<1>:d r39.0<1;1,0>:d r7.24<0;1,0>:uw {I@1}
0x89c0	0	macl (16|M0) r40.0<1>:d r39.0<1;1,0>:d r7.12<0;1,0>:d
0x89d0	0	(W) mul (16|M0) acc0.0<1>:ud r7.14<0;1,0>:ud r39.0<2;1,0>:uw
0x89e0	0	mach (16|M0) r41.0<1>:d r7.14<0;1,0>:ud r39.0<1;1,0>:ud
0x89f0	0	add (16|M0) r43.0<1>:d r39.0<1;1,0>:d r41.0<1;1,0>:d {I@1}
0x8a00	0	shr (16|M0) r46.0<1>:ud r43.0<1;1,0>:ud r7.15<0;1,0>:d {I@1}
0x8a10	0	(W) mul (16|M0) acc0.0<1>:d r46.0<1;1,0>:d r8.0<0;1,0>:uw {@1,$3.dst}
0x8a20	0	macl (16|M0) r49.0<1>:d r46.0<1;1,0>:d r8.0<0;1,0>:d
0x8a30	0	(W) mul (16|M0) acc0.0<1>:ud r8.2<0;1,0>:ud r46.0<2;1,0>:uw
0x8a40	0	mach (16|M0) r50.0<1>:d r8.2<0;1,0>:ud r46.0<1;1,0>:ud
0x8a50	0	add (16|M0) r52.0<1>:d r46.0<1;1,0>:d r50.0<1;1,0>:d {Compacted,I@1}
0x8a58	0	shr (16|M0) r53.0<1>:ud r52.0<1;1,0>:ud r8.3<0;1,0>:d {I@1}
0x8a68	0	(W) mov (2|M0) r2.0<1>:d r3.4<1;1,0>:d {Compacted,$1.dst}
0x8a70	0	add (16|M0) r51.0<1>:d r39.0<1;1,0>:d -r49.0<1;1,0>:d
0x8a80	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:d r8.8<0;1,0>:uw {I@3}
0x8a90	0	macl (16|M0) r54.0<1>:d r53.0<1;1,0>:d r8.4<0;1,0>:d {Compacted}
0x8a98	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r2.0<0;1,0>:uw {I@3}
0x8aa8	0	(W) mov (2|M0) r3.4<1>:d r3.6<1;1,0>:d
0x8ab8	0	add (16|M0) r56.0<1>:d r46.0<1;1,0>:d -r54.0<1;1,0>:d {Compacted,I@3}
0x8ac0	0	macl (16|M0) r55.0<1>:ud r51.0<1;1,0>:ud r2.0<0;1,0>:ud
0x8ad0	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r2.0<0;1,0>:uw
0x8ae0	0	mach (16|M0) r57.0<1>:d r51.0<1;1,0>:ud r2.0<0;1,0>:ud
0x8af0	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r2.2<0;1,0>:uw
0x8b00	0	macl (16|M0) r60.0<1>:d r51.0<1;1,0>:ud r2.1<0;1,0>:d
0x8b10	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r3.8<0;1,0>:uw {I@6}
0x8b20	0	(W) mov (2|M0) r8.5<1>:d r3.8<1;1,0>:d
0x8b30	0	macl (16|M0) r61.0<1>:ud r56.0<1;1,0>:ud r3.4<0;1,0>:ud
0x8b40	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r3.8<0;1,0>:uw
0x8b50	0	mach (16|M0) r62.0<1>:d r56.0<1;1,0>:ud r3.4<0;1,0>:ud
0x8b60	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r3.10<0;1,0>:uw
0x8b70	0	macl (16|M0) r70.0<1>:d r56.0<1;1,0>:ud r3.5<0;1,0>:d
0x8b80	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r8.10<0;1,0>:uw {I@6}
0x8b90	0	(W) mov (2|M0) r9.0<1>:d r3.12<1;1,0>:d {Compacted}
0x8b98	0	macl (16|M0) r71.0<1>:ud r53.0<1;1,0>:ud r8.5<0;1,0>:ud {Compacted}
0x8ba0	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r8.10<0;1,0>:uw
0x8bb0	0	mach (16|M0) r72.0<1>:d r53.0<1;1,0>:ud r8.5<0;1,0>:ud
0x8bc0	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r8.12<0;1,0>:uw
0x8bd0	0	macl (16|M0) r77.0<1>:d r53.0<1;1,0>:ud r8.6<0;1,0>:d
0x8be0	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r9.0<0;1,0>:uw {I@6}
0x8bf0	0	(W) mov (2|M0) r10.0<1>:d r3.14<1;1,0>:d {Compacted}
0x8bf8	0	macl (16|M0) r78.0<1>:ud r51.0<1;1,0>:ud r9.0<0;1,0>:ud {Compacted}
0x8c00	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r9.0<0;1,0>:uw
0x8c10	0	mach (16|M0) r79.0<1>:d r51.0<1;1,0>:ud r9.0<0;1,0>:ud
0x8c20	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r9.2<0;1,0>:uw
0x8c30	0	macl (16|M0) r84.0<1>:d r51.0<1;1,0>:ud r9.1<0;1,0>:d
0x8c40	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r10.0<0;1,0>:uw {I@6}
0x8c50	0	(W) mov (2|M0) r11.0<1>:f r4.0<1;1,0>:f {Compacted}
0x8c58	0	macl (16|M0) r85.0<1>:ud r56.0<1;1,0>:ud r10.0<0;1,0>:ud {Compacted}
0x8c60	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r10.0<0;1,0>:uw
0x8c70	0	mach (16|M0) r88.0<1>:d r56.0<1;1,0>:ud r10.0<0;1,0>:ud
0x8c80	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r10.2<0;1,0>:uw
0x8c90	0	macl (16|M0) r94.0<1>:d r56.0<1;1,0>:ud r10.1<0;1,0>:d
0x8ca0	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r11.0<0;1,0>:uw {F@1}
0x8cb0	0	add (16|M0) r57.0<1>:d r57.0<1;1,0>:d r60.0<1;1,0>:d {Compacted}
0x8cb8	0	add (16|M0) r79.0<1>:d r79.0<1;1,0>:d r84.0<1;1,0>:d {Compacted,I@7}
0x8cc0	0	macl (16|M0) r95.0<1>:ud r53.0<1;1,0>:ud r11.0<0;1,0>:ud {Compacted}
0x8cc8	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r11.0<0;1,0>:uw
0x8cd8	0	mach (16|M0) r96.0<1>:d r53.0<1;1,0>:ud r11.0<0;1,0>:ud
0x8ce8	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r11.2<0;1,0>:uw
0x8cf8	0	(W) mov (2|M0) r12.0<1>:d r4.4<1;1,0>:d {Compacted}
0x8d00	0	mov (16|M0) r58.0<2>:ud r55.0<1;1,0>:ud {Compacted}
0x8d08	0	add (16|M0) r62.0<1>:d r62.0<1;1,0>:d r70.0<1;1,0>:d {Compacted}
0x8d10	0	mov (16|M0) r80.0<2>:ud r78.0<1;1,0>:ud {Compacted}
0x8d18	0	add (16|M0) r88.0<1>:d r88.0<1;1,0>:d r94.0<1;1,0>:d {Compacted}
0x8d20	0	macl (16|M0) r101.0<1>:d r53.0<1;1,0>:ud r11.1<0;1,0>:d
0x8d30	0	mov (16|M0) r58.1<2>:d r57.0<1;1,0>:d
0x8d40	0	mov (16|M0) r80.1<2>:d r79.0<1;1,0>:d
0x8d50	0	add (16|M0) r42.0<1>:d r126.0<1;1,0>:d -r40.0<1;1,0>:d {Compacted}
0x8d58	0	mov (16|M0) r63.0<2>:ud r61.0<1;1,0>:ud {Compacted}
0x8d60	0	add (16|M0) r72.0<1>:d r72.0<1;1,0>:d r77.0<1;1,0>:d {Compacted}
0x8d68	0	mov (16|M0) r89.0<2>:ud r85.0<1;1,0>:ud {Compacted}
0x8d70	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r12.0<0;1,0>:uw
0x8d80	0	add (16|M0) r96.0<1>:d r96.0<1;1,0>:d r101.0<1;1,0>:d {Compacted,I@7}
0x8d88	0	add (16|M0) r68.0<1>:q r58.0<1;1,0>:q r3.1<0;1,0>:q {Compacted,I@7}
0x8d90	0	add (16|M0) r92.0<1>:q r80.0<1;1,0>:q r3.5<0;1,0>:q {Compacted,I@7}
0x8d98	0	mov (16|M0) r63.1<2>:d r62.0<1;1,0>:d
0x8da8	0	mov (16|M0) r89.1<2>:d r88.0<1;1,0>:d
0x8db8	0	(W) mov (2|M0) r13.0<1>:d r4.6<1;1,0>:d {Compacted}
0x8dc0	0	macl (16|M0) r102.0<1>:ud r51.0<1;1,0>:ud r12.0<0;1,0>:ud {Compacted}
0x8dc8	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r12.0<0;1,0>:uw
0x8dd8	0	mov (16|M0) r73.0<2>:ud r71.0<1;1,0>:ud {Compacted}
0x8de0	0	mov (16|M0) r97.0<2>:ud r95.0<1;1,0>:ud {Compacted}
0x8de8	0	mov (16|M0) r44.0<2>:ud r42.0<1;1,0>:ud {Compacted}
0x8df0	0	add (16|M0) r75.0<1>:q r68.0<1;1,0>:q r63.0<1;1,0>:q {Compacted,I@7}
0x8df8	0	add (16|M0) r99.0<1>:q r92.0<1;1,0>:q r89.0<1;1,0>:q {Compacted,I@7}
0x8e00	0	mach (16|M0) r103.0<1>:d r51.0<1;1,0>:ud r12.0<0;1,0>:ud
0x8e10	0	mov (16|M0) r73.1<2>:d r72.0<1;1,0>:d
0x8e20	0	mov (16|M0) r97.1<2>:d r96.0<1;1,0>:d
0x8e30	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r12.2<0;1,0>:uw
0x8e40	0	macl (16|M0) r108.0<1>:d r51.0<1;1,0>:ud r12.1<0;1,0>:d
0x8e50	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r13.0<0;1,0>:uw
0x8e60	0	(W) mov (2|M0) r14.0<1>:d r4.8<1;1,0>:d {Compacted}
0x8e68	0	shl (16|M0) r47.0<1>:q r44.0<2;1,0>:ud 2:w {I@7}
0x8e78	0	add (16|M0) r82.0<1>:q r75.0<1;1,0>:q r73.0<1;1,0>:q {Compacted,I@7}
0x8e80	0	add (16|M0) r106.0<1>:q r99.0<1;1,0>:q r97.0<1;1,0>:q {Compacted,I@7}
0x8e88	0	macl (16|M0) r109.0<1>:ud r56.0<1;1,0>:ud r13.0<0;1,0>:ud {Compacted}
0x8e90	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r13.0<0;1,0>:uw
0x8ea0	0	mach (16|M0) r112.0<1>:d r56.0<1;1,0>:ud r13.0<0;1,0>:ud
0x8eb0	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r13.2<0;1,0>:uw
0x8ec0	0	add (16|M0) r86.0<1>:q r82.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@6}
0x8ec8	0	add (16|M0) r110.0<1>:q r106.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@6}
0x8ed0	0	macl (16|M0) r118.0<1>:d r56.0<1;1,0>:ud r13.1<0;1,0>:d
0x8ee0	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r14.0<0;1,0>:uw
0x8ef0	0	add (16|M0) r103.0<1>:d r103.0<1;1,0>:d r108.0<1;1,0>:d {Compacted}
0x8ef8	0	macl (16|M0) r119.0<1>:ud r53.0<1;1,0>:ud r14.0<0;1,0>:ud {Compacted}
0x8f00	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r14.0<0;1,0>:uw
0x8f10	0	send.ugm (16|M0) r91 r86 null:0 0x0 0x04100580 {I@7,$4} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x8f20	0	send.ugm (16|M0) r115 r110 null:0 0x0 0x04100580 {I@6,$5} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x8f30	0	mach (16|M0) r120.0<1>:d r53.0<1;1,0>:ud r14.0<0;1,0>:ud
0x8f40	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r14.2<0;1,0>:uw
0x8f50	0	mov (16|M0) r104.0<2>:ud r102.0<1;1,0>:ud {Compacted}
0x8f58	0	add (16|M0) r112.0<1>:d r112.0<1;1,0>:d r118.0<1;1,0>:d {Compacted,I@7}
0x8f60	0	macl (16|M0) r10.0<1>:d r53.0<1;1,0>:ud r14.1<0;1,0>:d
0x8f70	0	mov (16|M0) r104.1<2>:d r103.0<1;1,0>:d {I@7}
0x8f80	0	mov (16|M0) r113.0<2>:ud r109.0<1;1,0>:ud {Compacted}
0x8f88	0	add (16|M0) r120.0<1>:d r120.0<1;1,0>:d r10.0<1;1,0>:d {Compacted,I@3}
0x8f90	0	add (16|M0) r116.0<1>:q r104.0<1;1,0>:q r4.1<0;1,0>:q {Compacted,I@3}
0x8f98	0	mov (16|M0) r113.1<2>:d r112.0<1;1,0>:d
0x8fa8	0	mov (16|M0) r121.0<2>:ud r119.0<1;1,0>:ud {Compacted}
0x8fb0	0	add (16|M0) r8.0<1>:q r116.0<1;1,0>:q r113.0<1;1,0>:q {I@2}
0x8fc0	0	mov (16|M0) r121.1<2>:d r120.0<1;1,0>:d
0x8fd0	0	add (16|M0) r38.0<1>:q r8.0<1;1,0>:q r121.0<1;1,0>:q {Compacted,I@1}
0x8fd8	0	add (16|M0) r42.0<1>:q r38.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@1}
0x8fe0	0	send.ugm (16|M0) r49 r42 null:0 0x0 0x04100580 {I@1,$6} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x8ff0	0	(W) mov (2|M0) r15.0<1>:d r4.12<1;1,0>:d {Compacted}
0x8ff8	0	(W) mov (2|M0) r16.0<1>:d r4.14<1;1,0>:d {Compacted}
0x9000	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r15.0<0;1,0>:uw {I@2}
0x9010	0	macl (16|M0) r11.0<1>:ud r51.0<1;1,0>:ud r15.0<0;1,0>:ud
0x9020	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r15.0<0;1,0>:uw
0x9030	0	mach (16|M0) r12.0<1>:d r51.0<1;1,0>:ud r15.0<0;1,0>:ud
0x9040	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r15.2<0;1,0>:uw
0x9050	0	macl (16|M0) r40.0<1>:d r51.0<1;1,0>:ud r15.1<0;1,0>:d
0x9060	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r16.0<0;1,0>:uw {I@7}
0x9070	0	(W) mov (2|M0) r17.0<1>:f r5.0<1;1,0>:f {Compacted}
0x9078	0	macl (16|M0) r41.0<1>:ud r56.0<1;1,0>:ud r16.0<0;1,0>:ud {Compacted}
0x9080	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r16.0<0;1,0>:uw
0x9090	0	mach (16|M0) r44.0<1>:d r56.0<1;1,0>:ud r16.0<0;1,0>:ud
0x90a0	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r16.2<0;1,0>:uw
0x90b0	0	macl (16|M0) r57.0<1>:d r56.0<1;1,0>:ud r16.1<0;1,0>:d
0x90c0	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r17.0<0;1,0>:uw {F@1}
0x90d0	0	add (16|M0) r12.0<1>:d r12.0<1;1,0>:d r40.0<1;1,0>:d {Compacted,I@7}
0x90d8	0	macl (16|M0) r58.0<1>:ud r53.0<1;1,0>:ud r17.0<0;1,0>:ud {Compacted}
0x90e0	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r17.0<0;1,0>:uw
0x90f0	0	(W) mov (2|M0) r18.0<1>:d r5.4<1;1,0>:d
0x9100	0	mach (16|M0) r59.0<1>:d r53.0<1;1,0>:ud r17.0<0;1,0>:ud
0x9110	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r17.2<0;1,0>:uw
0x9120	0	mov (16|M0) r13.0<2>:ud r11.0<1;1,0>:ud {Compacted}
0x9128	0	add (16|M0) r44.0<1>:d r44.0<1;1,0>:d r57.0<1;1,0>:d {Compacted,I@7}
0x9130	0	macl (16|M0) r64.0<1>:d r53.0<1;1,0>:ud r17.1<0;1,0>:d
0x9140	0	mov (16|M0) r13.1<2>:d r12.0<1;1,0>:d {I@7}
0x9150	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r18.0<0;1,0>:uw {I@7}
0x9160	0	(W) mov (2|M0) r20.0<1>:d r5.6<1;1,0>:d {Compacted}
0x9168	0	mov (16|M0) r45.0<2>:ud r41.0<1;1,0>:ud {Compacted}
0x9170	0	add (16|M0) r59.0<1>:d r59.0<1;1,0>:d r64.0<1;1,0>:d {Compacted,I@5}
0x9178	0	add (16|M0) r54.0<1>:q r13.0<1;1,0>:q r4.5<0;1,0>:q {Compacted,I@5}
0x9180	0	macl (16|M0) r68.0<1>:ud r51.0<1;1,0>:ud r18.0<0;1,0>:ud
0x9190	0	mov (16|M0) r45.1<2>:d r44.0<1;1,0>:d
0x91a0	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r18.0<0;1,0>:uw
0x91b0	0	mach (16|M0) r69.0<1>:d r51.0<1;1,0>:ud r18.0<0;1,0>:ud
0x91c0	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r18.2<0;1,0>:uw
0x91d0	0	mov (16|M0) r60.0<2>:ud r58.0<1;1,0>:ud {Compacted}
0x91d8	0	add (16|M0) r62.0<1>:q r54.0<1;1,0>:q r45.0<1;1,0>:q {Compacted,I@5}
0x91e0	0	macl (16|M0) r74.0<1>:d r51.0<1;1,0>:ud r18.1<0;1,0>:d
0x91f0	0	mov (16|M0) r60.1<2>:d r59.0<1;1,0>:d
0x9200	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r20.0<0;1,0>:uw
0x9210	0	(W) mov (2|M0) r21.0<1>:d r5.8<1;1,0>:d {Compacted}
0x9218	0	macl (16|M0) r75.0<1>:ud r56.0<1;1,0>:ud r20.0<0;1,0>:ud {Compacted}
0x9220	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r20.0<0;1,0>:uw
0x9230	0	sync.nop null {Compacted,$5.dst}
0x9238	0	add (16|M0) acc2.0<1>:f r91.0<1;1,0>:f r115.0<1;1,0>:f {Compacted,$4.dst}
0x9240	0	add (16|M0) r72.0<1>:q r62.0<1;1,0>:q r60.0<1;1,0>:q {I@5}
0x9250	0	mach (16|M0) r78.0<1>:d r56.0<1;1,0>:ud r20.0<0;1,0>:ud
0x9260	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r20.2<0;1,0>:uw
0x9270	0	macl (16|M0) r84.0<1>:d r56.0<1;1,0>:ud r20.1<0;1,0>:d
0x9280	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r21.0<0;1,0>:uw {I@7}
0x9290	0	add (16|M0) r69.0<1>:d r69.0<1;1,0>:d r74.0<1;1,0>:d {Compacted}
0x9298	0	add (16|M0) r76.0<1>:q r72.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@6}
0x92a0	0	macl (16|M0) r85.0<1>:ud r53.0<1;1,0>:ud r21.0<0;1,0>:ud {Compacted}
0x92a8	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r21.0<0;1,0>:uw
0x92b8	0	(W) mov (2|M0) r22.0<1>:d r5.12<1;1,0>:d {Compacted}
0x92c0	0	mach (16|M0) r86.0<1>:d r53.0<1;1,0>:ud r21.0<0;1,0>:ud
0x92d0	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r21.2<0;1,0>:uw
0x92e0	0	send.ugm (16|M0) r81 r76 null:0 0x0 0x04100580 {I@6,$7} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x92f0	0	mov (16|M0) r70.0<2>:ud r68.0<1;1,0>:ud {Compacted}
0x92f8	0	add (16|M0) r78.0<1>:d r78.0<1;1,0>:d r84.0<1;1,0>:d {Compacted}
0x9300	0	macl (16|M0) r91.0<1>:d r53.0<1;1,0>:ud r21.1<0;1,0>:d {F@1}
0x9310	0	mov (16|M0) r70.1<2>:d r69.0<1;1,0>:d
0x9320	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r22.0<0;1,0>:uw {I@7}
0x9330	0	(W) mov (2|M0) r23.0<1>:d r5.14<1;1,0>:d {Compacted}
0x9338	0	mov (16|M0) r79.0<2>:ud r75.0<1;1,0>:ud {Compacted}
0x9340	0	add (16|M0) r86.0<1>:d r86.0<1;1,0>:d r91.0<1;1,0>:d {Compacted,I@5}
0x9348	0	add (16|M0) r82.0<1>:q r70.0<1;1,0>:q r5.1<0;1,0>:q {Compacted,I@5}
0x9350	0	macl (16|M0) r92.0<1>:ud r51.0<1;1,0>:ud r22.0<0;1,0>:ud
0x9360	0	mov (16|M0) r79.1<2>:d r78.0<1;1,0>:d
0x9370	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r22.0<0;1,0>:uw
0x9380	0	mach (16|M0) r93.0<1>:d r51.0<1;1,0>:ud r22.0<0;1,0>:ud
0x9390	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r22.2<0;1,0>:uw
0x93a0	0	mov (16|M0) r87.0<2>:ud r85.0<1;1,0>:ud {Compacted}
0x93a8	0	add (16|M0) r89.0<1>:q r82.0<1;1,0>:q r79.0<1;1,0>:q {Compacted,I@5}
0x93b0	0	macl (16|M0) r98.0<1>:d r51.0<1;1,0>:ud r22.1<0;1,0>:d
0x93c0	0	mov (16|M0) r87.1<2>:d r86.0<1;1,0>:d
0x93d0	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r23.0<0;1,0>:uw
0x93e0	0	macl (16|M0) r99.0<1>:ud r56.0<1;1,0>:ud r23.0<0;1,0>:ud {Compacted}
0x93e8	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r23.0<0;1,0>:uw
0x93f8	0	(W) mov (2|M0) r24.0<1>:f r6.0<1;1,0>:f {Compacted}
0x9400	0	add (16|M0) r96.0<1>:q r89.0<1;1,0>:q r87.0<1;1,0>:q {Compacted,I@4}
0x9408	0	add (16|M0) acc2.0<1>:f acc2.0<1;1,0>:f r49.0<1;1,0>:f {Compacted,$6.dst}
0x9410	0	mach (16|M0) r102.0<1>:d r56.0<1;1,0>:ud r23.0<0;1,0>:ud
0x9420	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r23.2<0;1,0>:uw
0x9430	0	add (16|M0) r100.0<1>:q r96.0<1;1,0>:q r47.0<1;1,0>:q {I@3}
0x9440	0	macl (16|M0) r108.0<1>:d r56.0<1;1,0>:ud r23.1<0;1,0>:d
0x9450	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r24.0<0;1,0>:uw {F@2}
0x9460	0	add (16|M0) r93.0<1>:d r93.0<1;1,0>:d r98.0<1;1,0>:d {Compacted}
0x9468	0	macl (16|M0) r109.0<1>:ud r53.0<1;1,0>:ud r24.0<0;1,0>:ud {Compacted}
0x9470	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r24.0<0;1,0>:uw
0x9480	0	send.ugm (16|M0) r105 r100 null:0 0x0 0x04100580 {I@6,$8} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x9490	0	mach (16|M0) r110.0<1>:d r53.0<1;1,0>:ud r24.0<0;1,0>:ud
0x94a0	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r24.2<0;1,0>:uw
0x94b0	0	mov (16|M0) r94.0<2>:ud r92.0<1;1,0>:ud {Compacted}
0x94b8	0	add (16|M0) r102.0<1>:d r102.0<1;1,0>:d r108.0<1;1,0>:d {Compacted,I@7}
0x94c0	0	macl (16|M0) r115.0<1>:d r53.0<1;1,0>:ud r24.1<0;1,0>:d
0x94d0	0	mov (16|M0) r94.1<2>:d r93.0<1;1,0>:d {I@7}
0x94e0	0	mov (16|M0) r103.0<2>:ud r99.0<1;1,0>:ud {Compacted}
0x94e8	0	add (16|M0) r110.0<1>:d r110.0<1;1,0>:d r115.0<1;1,0>:d {Compacted,I@3}
0x94f0	0	add (16|M0) r106.0<1>:q r94.0<1;1,0>:q r5.5<0;1,0>:q {I@3}
0x9500	0	mov (16|M0) r103.1<2>:d r102.0<1;1,0>:d
0x9510	0	mov (16|M0) r111.0<2>:ud r109.0<1;1,0>:ud {Compacted}
0x9518	0	add (16|M0) r113.0<1>:q r106.0<1;1,0>:q r103.0<1;1,0>:q {Compacted,I@2}
0x9520	0	mov (16|M0) r111.1<2>:d r110.0<1;1,0>:d
0x9530	0	add (16|M0) r120.0<1>:q r113.0<1;1,0>:q r111.0<1;1,0>:q {Compacted,I@1}
0x9538	0	add (16|M0) r8.0<1>:q r120.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@1}
0x9540	0	send.ugm (16|M0) r13 r8 null:0 0x0 0x04100580 {I@1,$9} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x9550	0	(W) mov (2|M0) r25.0<1>:d r6.4<1;1,0>:d {Compacted}
0x9558	0	(W) mov (2|M0) r26.0<1>:d r6.6<1;1,0>:d {Compacted}
0x9560	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r25.0<0;1,0>:uw {I@2}
0x9570	0	macl (16|M0) r116.0<1>:ud r51.0<1;1,0>:ud r25.0<0;1,0>:ud
0x9580	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r25.0<0;1,0>:uw
0x9590	0	mach (16|M0) r117.0<1>:d r51.0<1;1,0>:ud r25.0<0;1,0>:ud
0x95a0	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r25.2<0;1,0>:uw
0x95b0	0	macl (16|M0) r1.0<1>:d r51.0<1;1,0>:ud r25.1<0;1,0>:d
0x95c0	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r26.0<0;1,0>:uw {I@7}
0x95d0	0	add (16|M0) acc2.0<1>:f acc2.0<1;1,0>:f r81.0<1;1,0>:f {Compacted,$7.dst}
0x95d8	0	(W) mov (2|M0) r27.0<1>:d r6.8<1;1,0>:d {Compacted}
0x95e0	0	macl (16|M0) r4.0<1>:ud r56.0<1;1,0>:ud r26.0<0;1,0>:ud
0x95f0	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r26.0<0;1,0>:uw
0x9600	0	mach (16|M0) r10.0<1>:d r56.0<1;1,0>:ud r26.0<0;1,0>:ud
0x9610	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r26.2<0;1,0>:uw
0x9620	0	macl (16|M0) r16.0<1>:d r56.0<1;1,0>:ud r26.1<0;1,0>:d
0x9630	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r27.0<0;1,0>:uw {I@6}
0x9640	0	(W) mov (2|M0) r28.0<1>:d r6.12<1;1,0>:d {Compacted}
0x9648	0	macl (16|M0) r17.0<1>:ud r53.0<1;1,0>:ud r27.0<0;1,0>:ud {Compacted}
0x9650	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r27.0<0;1,0>:uw
0x9660	0	mach (16|M0) r18.0<1>:d r53.0<1;1,0>:ud r27.0<0;1,0>:ud
0x9670	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r27.2<0;1,0>:uw
0x9680	0	macl (16|M0) r23.0<1>:d r53.0<1;1,0>:ud r27.1<0;1,0>:d
0x9690	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r28.0<0;1,0>:uw {I@6}
0x96a0	0	(W) mov (2|M0) r29.0<1>:d r6.14<1;1,0>:d {Compacted}
0x96a8	0	macl (16|M0) r24.0<1>:ud r51.0<1;1,0>:ud r28.0<0;1,0>:ud {Compacted}
0x96b0	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r28.0<0;1,0>:uw
0x96c0	0	mach (16|M0) r25.0<1>:d r51.0<1;1,0>:ud r28.0<0;1,0>:ud
0x96d0	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r28.2<0;1,0>:uw
0x96e0	0	macl (16|M0) r40.0<1>:d r51.0<1;1,0>:ud r28.1<0;1,0>:d
0x96f0	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r29.0<0;1,0>:uw {I@6}
0x9700	0	add (16|M0) acc2.0<1>:f acc2.0<1;1,0>:f r105.0<1;1,0>:f {Compacted,$8.dst}
0x9708	0	(W) mov (2|M0) r30.0<1>:f r7.0<1;1,0>:f {Compacted}
0x9710	0	macl (16|M0) r41.0<1>:ud r56.0<1;1,0>:ud r29.0<0;1,0>:ud
0x9720	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r29.0<0;1,0>:uw
0x9730	0	mach (16|M0) r44.0<1>:d r56.0<1;1,0>:ud r29.0<0;1,0>:ud
0x9740	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r29.2<0;1,0>:uw
0x9750	0	add (16|M0) r117.0<1>:d r117.0<1;1,0>:d r1.0<1;1,0>:d {Compacted}
0x9758	0	macl (16|M0) r57.0<1>:d r56.0<1;1,0>:ud r29.1<0;1,0>:d
0x9768	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r30.0<0;1,0>:uw {F@1}
0x9778	0	(W) mov (2|M0) r31.0<1>:d r7.4<1;1,0>:d {Compacted}
0x9780	0	macl (16|M0) r58.0<1>:ud r53.0<1;1,0>:ud r30.0<0;1,0>:ud {Compacted}
0x9788	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r30.0<0;1,0>:uw
0x9798	0	mov (16|M0) r118.0<2>:ud r116.0<1;1,0>:ud {Compacted}
0x97a0	0	add (16|M0) r10.0<1>:d r10.0<1;1,0>:d r16.0<1;1,0>:d
0x97b0	0	mach (16|M0) r59.0<1>:d r53.0<1;1,0>:ud r30.0<0;1,0>:ud
0x97c0	0	mov (16|M0) r118.1<2>:d r117.0<1;1,0>:d {I@7}
0x97d0	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r30.2<0;1,0>:uw
0x97e0	0	macl (16|M0) r64.0<1>:d r53.0<1;1,0>:ud r30.1<0;1,0>:d
0x97f0	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r31.0<0;1,0>:uw {I@7}
0x9800	0	(W) mov (1|M0) r32.0<1>:q r2.2<0;1,0>:q
0x9810	0	(W) mov (2|M0) r33.0<1>:d r7.6<1;1,0>:d {Compacted}
0x9818	0	mov (16|M0) r11.0<2>:ud r4.0<1;1,0>:ud {Compacted}
0x9820	0	add (16|M0) r18.0<1>:d r18.0<1;1,0>:d r23.0<1;1,0>:d {Compacted}
0x9828	0	add (16|M0) r14.0<1>:q r118.0<1;1,0>:q r6.1<0;1,0>:q {Compacted,I@7}
0x9830	0	macl (16|M0) r68.0<1>:ud r51.0<1;1,0>:ud r31.0<0;1,0>:ud
0x9840	0	mov (16|M0) r11.1<2>:d r10.0<1;1,0>:d
0x9850	0	(W) mul (16|M0) acc0.0<1>:ud r51.0<1;1,0>:ud r31.0<0;1,0>:uw
0x9860	0	mach (16|M0) r69.0<1>:d r51.0<1;1,0>:ud r31.0<0;1,0>:ud
0x9870	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r31.2<0;1,0>:uw
0x9880	0	(W) send.ugm (1|M0) r36 r32 null:0 0x38000 0x0210A780 {I@7,$10} [, msg-length:1, resp-length:1, header:no, func-control:a780]
0x9890	0	mov (16|M0) r19.0<2>:ud r17.0<1;1,0>:ud {Compacted}
0x9898	0	add (16|M0) r21.0<1>:q r14.0<1;1,0>:q r11.0<1;1,0>:q {Compacted,I@5}
0x98a0	0	macl (16|M0) r74.0<1>:d r51.0<1;1,0>:ud r31.1<0;1,0>:d
0x98b0	0	mov (16|M0) r19.1<2>:d r18.0<1;1,0>:d
0x98c0	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r33.0<0;1,0>:uw
0x98d0	0	macl (16|M0) r75.0<1>:ud r56.0<1;1,0>:ud r33.0<0;1,0>:ud {Compacted}
0x98d8	0	(W) mul (16|M0) acc0.0<1>:ud r56.0<1;1,0>:ud r33.0<0;1,0>:uw
0x98e8	0	(W) mov (2|M0) r35.0<1>:d r7.8<1;1,0>:d {Compacted}
0x98f0	0	add (16|M0) r38.0<1>:q r21.0<1;1,0>:q r19.0<1;1,0>:q {Compacted,I@5}
0x98f8	0	add (16|M0) acc2.0<1>:f acc2.0<1;1,0>:f r13.0<1;1,0>:f {Compacted,$9.dst}
0x9900	0	mach (16|M0) r78.0<1>:d r56.0<1;1,0>:ud r33.0<0;1,0>:ud
0x9910	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r33.2<0;1,0>:uw
0x9920	0	add (16|M0) r25.0<1>:d r25.0<1;1,0>:d r40.0<1;1,0>:d {Compacted}
0x9928	0	add (16|M0) r42.0<1>:q r38.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@4}
0x9930	0	macl (16|M0) r84.0<1>:d r56.0<1;1,0>:ud r33.1<0;1,0>:d
0x9940	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r35.0<0;1,0>:uw
0x9950	0	add (16|M0) r69.0<1>:d r69.0<1;1,0>:d r74.0<1;1,0>:d {Compacted}
0x9958	0	macl (16|M0) r85.0<1>:ud r53.0<1;1,0>:ud r35.0<0;1,0>:ud {Compacted}
0x9960	0	(W) mul (16|M0) acc0.0<1>:ud r53.0<1;1,0>:ud r35.0<0;1,0>:uw
0x9970	0	send.ugm (16|M0) r49 r42 null:0 0x0 0x04100580 {I@6,$11} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x9980	0	mov (16|M0) r26.0<2>:ud r24.0<1;1,0>:ud {Compacted}
0x9988	0	add (16|M0) r44.0<1>:d r44.0<1;1,0>:d r57.0<1;1,0>:d {Compacted}
0x9990	0	mach (16|M0) r86.0<1>:d r53.0<1;1,0>:ud r35.0<0;1,0>:ud
0x99a0	0	mov (16|M0) r26.1<2>:d r25.0<1;1,0>:d
0x99b0	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r35.2<0;1,0>:uw
0x99c0	0	mov (16|M0) r70.0<2>:ud r68.0<1;1,0>:ud {Compacted}
0x99c8	0	add (16|M0) r78.0<1>:d r78.0<1;1,0>:d r84.0<1;1,0>:d {Compacted,I@7}
0x99d0	0	macl (16|M0) r91.0<1>:d r53.0<1;1,0>:ud r35.1<0;1,0>:d
0x99e0	0	mov (16|M0) r70.1<2>:d r69.0<1;1,0>:d {I@7}
0x99f0	0	mov (16|M0) r45.0<2>:ud r41.0<1;1,0>:ud {Compacted}
0x99f8	0	add (16|M0) r59.0<1>:d r59.0<1;1,0>:d r64.0<1;1,0>:d {Compacted}
0x9a00	0	add (16|M0) r54.0<1>:q r26.0<1;1,0>:q r6.5<0;1,0>:q {Compacted,I@7}
0x9a08	0	mov (16|M0) r45.1<2>:d r44.0<1;1,0>:d
0x9a18	0	mov (16|M0) r79.0<2>:ud r75.0<1;1,0>:ud {Compacted}
0x9a20	0	add (16|M0) r86.0<1>:d r86.0<1;1,0>:d r91.0<1;1,0>:d {Compacted,I@7}
0x9a28	0	add (16|M0) r82.0<1>:q r70.0<1;1,0>:q r7.1<0;1,0>:q {Compacted,I@7}
0x9a30	0	mov (16|M0) r79.1<2>:d r78.0<1;1,0>:d
0x9a40	0	mov (16|M0) r60.0<2>:ud r58.0<1;1,0>:ud {Compacted}
0x9a48	0	add (16|M0) r62.0<1>:q r54.0<1;1,0>:q r45.0<1;1,0>:q {Compacted,I@6}
0x9a50	0	mov (16|M0) r60.1<2>:d r59.0<1;1,0>:d
0x9a60	0	mov (16|M0) r87.0<2>:ud r85.0<1;1,0>:ud {Compacted}
0x9a68	0	add (16|M0) r89.0<1>:q r82.0<1;1,0>:q r79.0<1;1,0>:q {Compacted,I@5}
0x9a70	0	mov (16|M0) r87.1<2>:d r86.0<1;1,0>:d
0x9a80	0	(W) send.ugm (1|M0) r37 r32 null:0 0xF8000 0x02108780 {$12} [, msg-length:1, resp-length:1, header:no, func-control:8780]
0x9a90	0	add (16|M0) r72.0<1>:q r62.0<1;1,0>:q r60.0<1;1,0>:q {Compacted,I@4}
0x9a98	0	add (16|M0) r96.0<1>:q r89.0<1;1,0>:q r87.0<1;1,0>:q {Compacted,I@2}
0x9aa0	0	add (16|M0) r76.0<1>:q r72.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@2}
0x9aa8	0	add (16|M0) r99.0<1>:q r96.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@2}
0x9ab0	0	send.ugm (16|M0) r81 r76 null:0 0x0 0x04100580 {I@2,$13} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x9ac0	0	send.ugm (16|M0) r102 r99 null:0 0x0 0x04100580 {I@1,$14} [, msg-length:2, resp-length:1, header:no, func-control:580]
0x9ad0	0	(W) mov (2|M0) r65.0<1>:f r36.0<1;1,0>:f {Compacted,$10.dst}
0x9ad8	0	(W) mov (2|M0) r66.0<1>:d r36.2<1;1,0>:d {Compacted}
0x9ae0	0	(W) mul (16|M0) acc0.0<1>:ud r65.0<0;1,0>:ud r51.0<2;1,0>:uw {F@1}
0x9af0	0	macl (16|M0) r92.0<1>:ud r65.0<0;1,0>:ud r51.0<1;1,0>:ud
0x9b00	0	(W) mul (16|M0) acc0.0<1>:ud r65.0<0;1,0>:ud r51.0<2;1,0>:uw
0x9b10	0	mach (16|M0) r93.0<1>:d r65.0<0;1,0>:ud r51.0<1;1,0>:ud
0x9b20	0	(W) mul (16|M0) acc0.0<1>:d r51.0<1;1,0>:ud r65.2<0;1,0>:uw
0x9b30	0	macl (16|M0) r98.0<1>:d r51.0<1;1,0>:ud r65.1<0;1,0>:d
0x9b40	0	(W) mul (16|M0) acc0.0<1>:ud r66.0<0;1,0>:ud r56.0<2;1,0>:uw {I@7}
0x9b50	0	add (16|M0) acc2.0<1>:f acc2.0<1;1,0>:f r49.0<1;1,0>:f {Compacted,$11.dst}
0x9b58	0	(W) mov (2|M0) r67.0<1>:d r36.4<1;1,0>:d {Compacted}
0x9b60	0	macl (16|M0) r101.0<1>:ud r66.0<0;1,0>:ud r56.0<1;1,0>:ud
0x9b70	0	(W) mul (16|M0) acc0.0<1>:ud r66.0<0;1,0>:ud r56.0<2;1,0>:uw
0x9b80	0	mach (16|M0) r103.0<1>:d r66.0<0;1,0>:ud r56.0<1;1,0>:ud
0x9b90	0	(W) mul (16|M0) acc0.0<1>:d r56.0<1;1,0>:ud r66.2<0;1,0>:uw
0x9ba0	0	macl (16|M0) r108.0<1>:d r56.0<1;1,0>:ud r66.1<0;1,0>:d
0x9bb0	0	(W) mul (16|M0) acc0.0<1>:ud r67.0<0;1,0>:ud r53.0<2;1,0>:uw {I@6}
0x9bc0	0	mov (16|M0) r94.0<2>:ud r92.0<1;1,0>:ud {Compacted}
0x9bc8	0	macl (16|M0) r109.0<1>:ud r67.0<0;1,0>:ud r53.0<1;1,0>:ud {Compacted}
0x9bd0	0	add (16|M0) r94.1<2>:d r93.0<1;1,0>:d r98.0<1;1,0>:d
0x9be0	0	(W) mul (16|M0) acc0.0<1>:ud r67.0<0;1,0>:ud r53.0<2;1,0>:uw
0x9bf0	0	mach (16|M0) r110.0<1>:d r67.0<0;1,0>:ud r53.0<1;1,0>:ud
0x9c00	0	(W) mul (16|M0) acc0.0<1>:d r53.0<1;1,0>:ud r67.2<0;1,0>:uw
0x9c10	0	mov (16|M0) r104.0<2>:ud r101.0<1;1,0>:ud {Compacted}
0x9c18	0	add (16|M0) r106.0<1>:q r94.0<1;1,0>:q r37.0<0;1,0>:q {Compacted,@5,$12.dst}
0x9c20	0	macl (16|M0) r115.0<1>:d r53.0<1;1,0>:ud r67.1<0;1,0>:d
0x9c30	0	add (16|M0) r104.1<2>:d r103.0<1;1,0>:d r108.0<1;1,0>:d
0x9c40	0	mov (16|M0) r111.0<2>:ud r109.0<1;1,0>:ud {Compacted}
0x9c48	0	add (16|M0) r113.0<1>:q r106.0<1;1,0>:q r104.0<1;1,0>:q {Compacted,I@2}
0x9c50	0	add (16|M0) r111.1<2>:d r110.0<1;1,0>:d r115.0<1;1,0>:d
0x9c60	0	add (16|M0) acc0.0<1>:f acc2.0<1;1,0>:f r81.0<1;1,0>:f {Compacted,$13.dst}
0x9c68	0	add (16|M0) r116.0<1>:q r113.0<1;1,0>:q r111.0<1;1,0>:q {Compacted,I@1}
0x9c70	0	add (16|M0) acc0.0<1>:f acc0.0<1;1,0>:f r102.0<1;1,0>:f {Compacted,$14.dst}
0x9c78	0	add (16|M0) r118.0<1>:q r116.0<1;1,0>:q r47.0<1;1,0>:q {Compacted,I@1}
0x9c80	0	mad (16|M0) r120.0<1>:f r3.0<0;0>:f acc0.0<1;0>:f r2.7<0>:f {Compacted}
0x9c88	0	send.ugm (16|M0) null r118 r120:1 0x0 0x04000584 {A@1,$15} [, msg-length:2, resp-length:0, header:no, func-control:584]
0x9c98	0	Block 3:
0x9c98	0	join (16|M0) bb_4
0x9ca8	0	Block 4:
0x9ca8	0	sync.allrd ($2,$3)
0x9cb8	0	(W) mov (16|M0) r127.0<1>:f r123.0<1;1,0>:f {Compacted,$1.src}
0x9cc0	0	(W) send.gtwy (1|M0) null r127 null:0 0x0 0x02000010 {EOT,F@1,$0} [, msg-length:1, resp-length:0, header:no, func-control:10]
0x9cd0	0	(W) mov (16|M0) null<1>:ud 0x4A1FFD37:ud
0x9ce0	0	(W) mov (16|M0) null<1>:ud 0x317CDC71:ud
0x9cf0	0	(W) mov (16|M0) null<1>:ud 0x0:ud
0x9d00	0	(W) mov (16|M0) null<1>:ud 0x12:ud
0x9d10	0	Block 5:
