

i
m
p
o
r
t
 
n
u
m
p
y
 
a
s
 
n
p


i
m
p
o
r
t
 
m
a
t
h




c
l
a
s
s
 
O
p
t
i
m
i
z
e
r
:


 
 
 
 
"
"
"
N
o
i
s
y
 
c
o
n
t
i
n
u
o
u
s
 
b
l
a
c
k
-
b
o
x
 
o
p
t
i
m
i
z
e
r
 
f
o
r
 
1
0
D
 
p
r
o
b
l
e
m
s
 
w
i
t
h
 
1
2
0
 
q
u
e
r
y
 
b
u
d
g
e
t
.
"
"
"


 
 
 
 


 
 
 
 
d
e
f
 
_
_
i
n
i
t
_
_
(
s
e
l
f
,
 
d
i
m
,
 
l
o
w
e
r
,
 
u
p
p
e
r
,
 
b
u
d
g
e
t
,
 
s
e
e
d
,
 
r
n
g
)
:


 
 
 
 
 
 
 
 
s
e
l
f
.
d
i
m
 
=
 
d
i
m


 
 
 
 
 
 
 
 
s
e
l
f
.
l
o
w
e
r
 
=
 
n
p
.
a
s
a
r
r
a
y
(
l
o
w
e
r
,
 
d
t
y
p
e
=
n
p
.
f
l
o
a
t
6
4
)


 
 
 
 
 
 
 
 
s
e
l
f
.
u
p
p
e
r
 
=
 
n
p
.
a
s
a
r
r
a
y
(
u
p
p
e
r
,
 
d
t
y
p
e
=
n
p
.
f
l
o
a
t
6
4
)


 
 
 
 
 
 
 
 
s
e
l
f
.
b
u
d
g
e
t
 
=
 
b
u
d
g
e
t


 
 
 
 
 
 
 
 
s
e
l
f
.
s
e
e
d
 
=
 
s
e
e
d


 
 
 
 
 
 
 
 
s
e
l
f
.
r
n
g
 
=
 
r
n
g


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
#
 
S
t
a
t
e


 
 
 
 
 
 
 
 
s
e
l
f
.
X
 
=
 
n
p
.
e
m
p
t
y
(
(
0
,
 
d
i
m
)
,
 
d
t
y
p
e
=
n
p
.
f
l
o
a
t
6
4
)


 
 
 
 
 
 
 
 
s
e
l
f
.
y
 
=
 
n
p
.
e
m
p
t
y
(
(
0
,
)
,
 
d
t
y
p
e
=
n
p
.
f
l
o
a
t
6
4
)


 
 
 
 
 
 
 
 
s
e
l
f
.
b
e
s
t
_
x
 
=
 
N
o
n
e


 
 
 
 
 
 
 
 
s
e
l
f
.
b
e
s
t
_
y
 
=
 
n
p
.
i
n
f


 
 
 
 
 
 
 
 
s
e
l
f
.
q
u
e
r
i
e
s
_
u
s
e
d
 
=
 
0


 
 
 
 
 
 
 
 
s
e
l
f
.
b
a
t
c
h
 
=
 
1


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
#
 
A
d
a
p
t
i
v
e
 
s
t
e
p
 
s
i
z
e


 
 
 
 
 
 
 
 
s
e
l
f
.
s
t
e
p
_
s
i
z
e
 
=
 
0
.
5
 
*
 
(
u
p
p
e
r
[
0
]
 
-
 
l
o
w
e
r
[
0
]
)


 
 
 
 
 
 
 
 
s
e
l
f
.
s
t
e
p
_
d
e
c
a
y
 
=
 
0
.
9
9


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
#
 
G
e
n
e
r
a
t
e
 
i
n
i
t
i
a
l
 
d
e
s
i
g
n


 
 
 
 
 
 
 
 
s
e
l
f
.
_
g
e
n
e
r
a
t
e
_
i
n
i
t
i
a
l
_
d
e
s
i
g
n
(
)


 
 
 
 


 
 
 
 
d
e
f
 
_
g
e
n
e
r
a
t
e
_
i
n
i
t
i
a
l
_
d
e
s
i
g
n
(
s
e
l
f
)
:


 
 
 
 
 
 
 
 
"
"
"
G
e
n
e
r
a
t
e
 
i
n
i
t
i
a
l
 
p
o
i
n
t
s
 
u
s
i
n
g
 
L
a
t
i
n
 
H
y
p
e
r
c
u
b
e
 
S
a
m
p
l
i
n
g
.
"
"
"


 
 
 
 
 
 
 
 
n
_
i
n
i
t
i
a
l
 
=
 
m
i
n
(
2
0
,
 
m
a
x
(
s
e
l
f
.
b
u
d
g
e
t
 
/
/
 
3
,
 
s
e
l
f
.
d
i
m
 
+
 
2
)
)


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
p
o
i
n
t
s
 
=
 
n
p
.
z
e
r
o
s
(
(
n
_
i
n
i
t
i
a
l
,
 
s
e
l
f
.
d
i
m
)
,
 
d
t
y
p
e
=
n
p
.
f
l
o
a
t
6
4
)


 
 
 
 
 
 
 
 
f
o
r
 
j
 
i
n
 
r
a
n
g
e
(
s
e
l
f
.
d
i
m
)
:


 
 
 
 
 
 
 
 
 
 
 
 
p
e
r
m
 
=
 
s
e
l
f
.
r
n
g
.
p
e
r
m
u
t
a
t
i
o
n
(
n
_
i
n
i
t
i
a
l
)


 
 
 
 
 
 
 
 
 
 
 
 
p
o
i
n
t
s
[
:
,
 
j
]
 
=
 
(
p
e
r
m
 
+
 
s
e
l
f
.
r
n
g
.
u
n
i
f
o
r
m
(
0
,
 
1
,
 
n
_
i
n
i
t
i
a
l
)
)
 
/
 
n
_
i
n
i
t
i
a
l


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
s
e
l
f
.
l
h
d
_
p
o
i
n
t
s
 
=
 
s
e
l
f
.
l
o
w
e
r
 
+
 
p
o
i
n
t
s
 
*
 
(
s
e
l
f
.
u
p
p
e
r
 
-
 
s
e
l
f
.
l
o
w
e
r
)


 
 
 
 


 
 
 
 
d
e
f
 
a
s
k
(
s
e
l
f
,
 
n
)
:


 
 
 
 
 
 
 
 
"
"
"
R
e
t
u
r
n
 
n
 
q
u
e
r
y
 
p
o
i
n
t
s
 
w
i
t
h
i
n
 
b
o
u
n
d
s
.
"
"
"


 
 
 
 
 
 
 
 
n
 
=
 
i
n
t
(
n
)


 
 
 
 
 
 
 
 
i
f
 
n
 
<
 
1
:


 
 
 
 
 
 
 
 
 
 
 
 
n
 
=
 
1


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
p
o
i
n
t
s
 
=
 
n
p
.
e
m
p
t
y
(
(
n
,
 
s
e
l
f
.
d
i
m
)
,
 
d
t
y
p
e
=
n
p
.
f
l
o
a
t
6
4
)


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
f
o
r
 
i
 
i
n
 
r
a
n
g
e
(
n
)
:


 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
s
e
l
f
.
q
u
e
r
i
e
s
_
u
s
e
d
 
>
=
 
s
e
l
f
.
b
u
d
g
e
t
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
#
 
B
u
d
g
e
t
 
e
x
h
a
u
s
t
e
d
 
-
 
r
e
t
u
r
n
 
b
e
s
t
 
k
n
o
w
n
 
p
o
i
n
t


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
s
e
l
f
.
b
e
s
t
_
x
 
i
s
 
n
o
t
 
N
o
n
e
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
o
i
n
t
s
[
i
]
 
=
 
s
e
l
f
.
b
e
s
t
_
x


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
e
l
s
e
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
o
i
n
t
s
[
i
]
 
=
 
s
e
l
f
.
l
o
w
e
r
 
+
 
s
e
l
f
.
r
n
g
.
u
n
i
f
o
r
m
(
0
,
 
1
,
 
s
e
l
f
.
d
i
m
)
 
*
 
(
s
e
l
f
.
u
p
p
e
r
 
-
 
s
e
l
f
.
l
o
w
e
r
)


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
c
o
n
t
i
n
u
e


 
 
 
 
 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
 
 
 
 
#
 
P
h
a
s
e
 
1
:
 
I
n
i
t
i
a
l
 
e
x
p
l
o
r
a
t
i
o
n
 
v
i
a
 
L
H
D


 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
s
e
l
f
.
q
u
e
r
i
e
s
_
u
s
e
d
 
<
 
l
e
n
(
s
e
l
f
.
l
h
d
_
p
o
i
n
t
s
)
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
o
i
n
t
s
[
i
]
 
=
 
s
e
l
f
.
l
h
d
_
p
o
i
n
t
s
[
s
e
l
f
.
q
u
e
r
i
e
s
_
u
s
e
d
]


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
q
u
e
r
i
e
s
_
u
s
e
d
 
+
=
 
1


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
c
o
n
t
i
n
u
e


 
 
 
 
 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
 
 
 
 
#
 
P
h
a
s
e
 
2
:
 
L
o
c
a
l
 
o
p
t
i
m
i
z
a
t
i
o
n
 
a
r
o
u
n
d
 
b
e
s
t
 
p
o
i
n
t


 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
s
e
l
f
.
b
e
s
t
_
x
 
i
s
 
n
o
t
 
N
o
n
e
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
#
 
A
d
a
p
t
i
v
e
 
p
e
r
t
u
r
b
a
t
i
o
n


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
r
o
g
r
e
s
s
 
=
 
s
e
l
f
.
q
u
e
r
i
e
s
_
u
s
e
d
 
/
 
s
e
l
f
.
b
u
d
g
e
t


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
t
e
p
 
=
 
s
e
l
f
.
s
t
e
p
_
s
i
z
e
 
*
 
(
1
.
0
 
-
 
p
r
o
g
r
e
s
s
)
 
*
*
 
0
.
5


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
#
 
G
e
n
e
r
a
t
e
 
c
a
n
d
i
d
a
t
e


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
e
r
t
u
r
b
a
t
i
o
n
 
=
 
s
e
l
f
.
r
n
g
.
n
o
r
m
a
l
(
0
,
 
s
t
e
p
,
 
s
e
l
f
.
d
i
m
)


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
c
a
n
d
i
d
a
t
e
 
=
 
s
e
l
f
.
b
e
s
t
_
x
 
+
 
p
e
r
t
u
r
b
a
t
i
o
n


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
c
a
n
d
i
d
a
t
e
 
=
 
n
p
.
c
l
i
p
(
c
a
n
d
i
d
a
t
e
,
 
s
e
l
f
.
l
o
w
e
r
,
 
s
e
l
f
.
u
p
p
e
r
)


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
o
i
n
t
s
[
i
]
 
=
 
c
a
n
d
i
d
a
t
e


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
q
u
e
r
i
e
s
_
u
s
e
d
 
+
=
 
1


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
c
o
n
t
i
n
u
e


 
 
 
 
 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
 
 
 
 
#
 
F
a
l
l
b
a
c
k
:
 
r
a
n
d
o
m
 
p
o
i
n
t


 
 
 
 
 
 
 
 
 
 
 
 
p
o
i
n
t
s
[
i
]
 
=
 
s
e
l
f
.
l
o
w
e
r
 
+
 
s
e
l
f
.
r
n
g
.
u
n
i
f
o
r
m
(
0
,
 
1
,
 
s
e
l
f
.
d
i
m
)
 
*
 
(
s
e
l
f
.
u
p
p
e
r
 
-
 
s
e
l
f
.
l
o
w
e
r
)


 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
q
u
e
r
i
e
s
_
u
s
e
d
 
+
=
 
1


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
p
o
i
n
t
s


 
 
 
 


 
 
 
 
d
e
f
 
t
e
l
l
(
s
e
l
f
,
 
X
,
 
y
,
 
m
e
t
a
d
a
t
a
=
N
o
n
e
)
:


 
 
 
 
 
 
 
 
"
"
"
U
p
d
a
t
e
 
m
o
d
e
l
 
w
i
t
h
 
o
b
s
e
r
v
e
d
 
p
o
i
n
t
s
.
"
"
"


 
 
 
 
 
 
 
 
X
 
=
 
n
p
.
a
s
a
r
r
a
y
(
X
,
 
d
t
y
p
e
=
n
p
.
f
l
o
a
t
6
4
)


 
 
 
 
 
 
 
 
y
 
=
 
n
p
.
a
s
a
r
r
a
y
(
y
,
 
d
t
y
p
e
=
n
p
.
f
l
o
a
t
6
4
)


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
i
f
 
X
.
n
d
i
m
 
=
=
 
1
:


 
 
 
 
 
 
 
 
 
 
 
 
X
 
=
 
X
.
r
e
s
h
a
p
e
(
1
,
 
-
1
)


 
 
 
 
 
 
 
 
 
 
 
 
y
 
=
 
n
p
.
a
r
r
a
y
(
[
y
]
)


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
f
o
r
 
i
 
i
n
 
r
a
n
g
e
(
l
e
n
(
X
)
)
:


 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
X
 
=
 
n
p
.
v
s
t
a
c
k
(
[
s
e
l
f
.
X
,
 
X
[
i
:
i
+
1
]
]
)


 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
y
 
=
 
n
p
.
a
p
p
e
n
d
(
s
e
l
f
.
y
,
 
y
[
i
]
)


 
 
 
 
 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
y
[
i
]
 
<
 
s
e
l
f
.
b
e
s
t
_
y
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
b
e
s
t
_
y
 
=
 
y
[
i
]


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
b
e
s
t
_
x
 
=
 
X
[
i
]
.
c
o
p
y
(
)


 
 
 
 
 
 
 
 


 
 
 
 
 
 
 
 
#
 
A
d
a
p
t
i
v
e
 
s
t
e
p
 
s
i
z
e
 
a
d
j
u
s
t
m
e
n
t


 
 
 
 
 
 
 
 
i
f
 
l
e
n
(
s
e
l
f
.
X
)
 
>
=
 
5
:


 
 
 
 
 
 
 
 
 
 
 
 
r
e
c
e
n
t
 
=
 
s
e
l
f
.
y
[
-
5
:
]


 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
l
e
n
(
r
e
c
e
n
t
)
 
>
 
1
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
i
m
p
r
o
v
e
m
e
n
t
 
=
 
r
e
c
e
n
t
[
0
]
 
-
 
r
e
c
e
n
t
[
-
1
]


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
i
m
p
r
o
v
e
m
e
n
t
 
>
 
0
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
s
t
e
p
_
s
i
z
e
 
*
=
 
s
e
l
f
.
s
t
e
p
_
d
e
c
a
y


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
e
l
s
e
:


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
s
t
e
p
_
s
i
z
e
 
*
=
 
(
2
.
0
 
-
 
s
e
l
f
.
s
t
e
p
_
d
e
c
a
y
)


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
e
l
f
.
s
t
e
p
_
s
i
z
e
 
=
 
n
p
.
c
l
i
p
(
s
e
l
f
.
s
t
e
p
_
s
i
z
e
,
 
0
.
0
1
,
 
1
.
0
)


