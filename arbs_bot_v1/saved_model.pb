£È#
î
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58!
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
:*
dtype0

Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/v/dense_2/kernel

)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes

:*
dtype0

Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/m/dense_2/kernel

)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes

:*
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:*
dtype0

Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/v/dense_1/kernel

)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes

: *
dtype0

Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/m/dense_1/kernel

)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes

: *
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
: *
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
: *
dtype0

Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	 *
dtype0

Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	 *
dtype0

Adam/v/gru_1/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/v/gru_1/gru_cell_1/bias

0Adam/v/gru_1/gru_cell_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/gru_1/gru_cell_1/bias*
_output_shapes
:	*
dtype0

Adam/m/gru_1/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/m/gru_1/gru_cell_1/bias

0Adam/m/gru_1/gru_cell_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/gru_1/gru_cell_1/bias*
_output_shapes
:	*
dtype0
®
(Adam/v/gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(Adam/v/gru_1/gru_cell_1/recurrent_kernel
§
<Adam/v/gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp(Adam/v/gru_1/gru_cell_1/recurrent_kernel* 
_output_shapes
:
*
dtype0
®
(Adam/m/gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(Adam/m/gru_1/gru_cell_1/recurrent_kernel
§
<Adam/m/gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp(Adam/m/gru_1/gru_cell_1/recurrent_kernel* 
_output_shapes
:
*
dtype0

Adam/v/gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/v/gru_1/gru_cell_1/kernel

2Adam/v/gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/gru_1/gru_cell_1/kernel* 
_output_shapes
:
*
dtype0

Adam/m/gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/m/gru_1/gru_cell_1/kernel

2Adam/m/gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/gru_1/gru_cell_1/kernel* 
_output_shapes
:
*
dtype0

Adam/v/gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/v/gru/gru_cell/bias

,Adam/v/gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/gru/gru_cell/bias*
_output_shapes
:	*
dtype0

Adam/m/gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/m/gru/gru_cell/bias

,Adam/m/gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/gru/gru_cell/bias*
_output_shapes
:	*
dtype0
¦
$Adam/v/gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$Adam/v/gru/gru_cell/recurrent_kernel

8Adam/v/gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp$Adam/v/gru/gru_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0
¦
$Adam/m/gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$Adam/m/gru/gru_cell/recurrent_kernel

8Adam/m/gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp$Adam/m/gru/gru_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

Adam/v/gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/v/gru/gru_cell/kernel

.Adam/v/gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/gru/gru_cell/kernel*
_output_shapes
:	*
dtype0

Adam/m/gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/m/gru/gru_cell/kernel

.Adam/m/gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/gru/gru_cell/kernel*
_output_shapes
:	*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

gru_1/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_namegru_1/gru_cell_1/bias

)gru_1/gru_cell_1/bias/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/bias*
_output_shapes
:	*
dtype0
 
!gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!gru_1/gru_cell_1/recurrent_kernel

5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_1/recurrent_kernel* 
_output_shapes
:
*
dtype0

gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namegru_1/gru_cell_1/kernel

+gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/kernel* 
_output_shapes
:
*
dtype0

gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namegru/gru_cell/bias
x
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes
:	*
dtype0

gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_namegru/gru_cell/recurrent_kernel

1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	 *
dtype0

serving_default_input_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
º
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1gru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru_1/gru_cell_1/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_507648

NoOpNoOp
ÐR
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*R
valueRBþQ B÷Q

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell
 
state_spec*
¦
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias*
¦
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
¦
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
Z
90
:1
;2
<3
=4
>5
'6
(7
/8
09
710
811*
Z
90
:1
;2
<3
=4
>5
'6
(7
/8
09
710
811*
* 
°
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_3* 
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
* 

L
_variables
M_iterations
N_learning_rate
O_index_dict
P
_momentums
Q_velocities
R_update_step_xla*

Sserving_default* 

90
:1
;2*

90
:1
;2*
* 


Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ztrace_0
[trace_1
\trace_2
]trace_3* 
6
^trace_0
_trace_1
`trace_2
atrace_3* 
* 
Ó
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator

9kernel
:recurrent_kernel
;bias*
* 

<0
=1
>2*

<0
=1
>2*
* 


istates
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
otrace_0
ptrace_1
qtrace_2
rtrace_3* 
6
strace_0
ttrace_1
utrace_2
vtrace_3* 
* 
Ó
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
}_random_generator

<kernel
=recurrent_kernel
>bias*
* 

'0
(1*

'0
(1*
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEgru/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEgru/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_1/gru_cell_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_1/gru_cell_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_1/gru_cell_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ú
M0
1
2
3
4
5
6
7
8
9
10
11
 12
¡13
¢14
£15
¤16
¥17
¦18
§19
¨20
©21
ª22
«23
¬24*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
f
0
1
2
3
4
5
¡6
£7
¥8
§9
©10
«11*
f
0
1
2
3
4
 5
¢6
¤7
¦8
¨9
ª10
¬11*
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

90
:1
;2*

90
:1
;2*
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

²trace_0
³trace_1* 

´trace_0
µtrace_1* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

<0
=1
>2*

<0
=1
>2*
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

»trace_0
¼trace_1* 

½trace_0
¾trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
¿	variables
À	keras_api

Átotal

Âcount*
M
Ã	variables
Ä	keras_api

Åtotal

Æcount
Ç
_fn_kwargs*
e_
VARIABLE_VALUEAdam/m/gru/gru_cell/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/gru/gru_cell/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/gru/gru_cell/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/gru/gru_cell/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/gru/gru_cell/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/gru/gru_cell/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/gru_1/gru_cell_1/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/gru_1/gru_cell_1/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/gru_1/gru_cell_1/recurrent_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/gru_1/gru_cell_1/recurrent_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/gru_1/gru_cell_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/gru_1/gru_cell_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Á0
Â1*

¿	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Å0
Æ1*

Ã	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¿
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOp+gru_1/gru_cell_1/kernel/Read/ReadVariableOp5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOp)gru_1/gru_cell_1/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp.Adam/m/gru/gru_cell/kernel/Read/ReadVariableOp.Adam/v/gru/gru_cell/kernel/Read/ReadVariableOp8Adam/m/gru/gru_cell/recurrent_kernel/Read/ReadVariableOp8Adam/v/gru/gru_cell/recurrent_kernel/Read/ReadVariableOp,Adam/m/gru/gru_cell/bias/Read/ReadVariableOp,Adam/v/gru/gru_cell/bias/Read/ReadVariableOp2Adam/m/gru_1/gru_cell_1/kernel/Read/ReadVariableOp2Adam/v/gru_1/gru_cell_1/kernel/Read/ReadVariableOp<Adam/m/gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOp<Adam/v/gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOp0Adam/m/gru_1/gru_cell_1/bias/Read/ReadVariableOp0Adam/v/gru_1/gru_cell_1/bias/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOp)Adam/m/dense_1/kernel/Read/ReadVariableOp)Adam/v/dense_1/kernel/Read/ReadVariableOp'Adam/m/dense_1/bias/Read/ReadVariableOp'Adam/v/dense_1/bias/Read/ReadVariableOp)Adam/m/dense_2/kernel/Read/ReadVariableOp)Adam/v/dense_2/kernel/Read/ReadVariableOp'Adam/m/dense_2/bias/Read/ReadVariableOp'Adam/v/dense_2/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*7
Tin0
.2,	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_510094
ò	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kernelgru_1/gru_cell_1/bias	iterationlearning_rateAdam/m/gru/gru_cell/kernelAdam/v/gru/gru_cell/kernel$Adam/m/gru/gru_cell/recurrent_kernel$Adam/v/gru/gru_cell/recurrent_kernelAdam/m/gru/gru_cell/biasAdam/v/gru/gru_cell/biasAdam/m/gru_1/gru_cell_1/kernelAdam/v/gru_1/gru_cell_1/kernel(Adam/m/gru_1/gru_cell_1/recurrent_kernel(Adam/v/gru_1/gru_cell_1/recurrent_kernelAdam/m/gru_1/gru_cell_1/biasAdam/v/gru_1/gru_cell_1/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biastotal_1count_1totalcount*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_510230üÏ
¨
¸
$__inference_gru_layer_call_fn_508376
inputs_0
unknown:	
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_506289}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
Ü
ª
while_cond_508461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_508461___redundant_placeholder04
0while_while_cond_508461___redundant_placeholder14
0while_while_cond_508461___redundant_placeholder24
0while_while_cond_508461___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¼

Ü
+__inference_gru_cell_1_layer_call_fn_509867

inputs
states_0
unknown:	
	unknown_0:

	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_506513p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0
À

(__inference_dense_2_layer_call_fn_509723

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_507013o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í=

while_body_509119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_1_readvariableop_resource_0:	E
1while_gru_cell_1_matmul_readvariableop_resource_0:
G
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_1_readvariableop_resource:	C
/while_gru_cell_1_matmul_readvariableop_resource:
E
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ë
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

×
D__inference_gru_cell_layer_call_and_return_conditional_losses_506173

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¼

Ü
+__inference_gru_cell_1_layer_call_fn_509853

inputs
states_0
unknown:	
	unknown_0:

	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_506368p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0
¦@
ç
gru_while_body_507770$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	F
3gru_while_gru_cell_matmul_readvariableop_resource_0:	I
5gru_while_gru_cell_matmul_1_readvariableop_resource_0:

gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	D
1gru_while_gru_cell_matmul_readvariableop_resource:	G
3gru_while_gru_cell_matmul_1_readvariableop_resource:
¢(gru/while/gru_cell/MatMul/ReadVariableOp¢*gru/while/gru_cell/MatMul_1/ReadVariableOp¢!gru/while/gru_cell/ReadVariableOp
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   º
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¾
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿß
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¢
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¥
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿo
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒQ
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: 
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: 
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0^gru/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
gru/while/NoOpNoOp)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"¸
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¼
¢
while_body_506043
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_506065_0:	*
while_gru_cell_506067_0:	+
while_gru_cell_506069_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_506065:	(
while_gru_cell_506067:	)
while_gru_cell_506069:
¢&while/gru_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0÷
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_506065_0while_gru_cell_506067_0while_gru_cell_506069_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_506030Ø
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "0
while_gru_cell_506065while_gru_cell_506065_0"0
while_gru_cell_506067while_gru_cell_506067_0"0
while_gru_cell_506069while_gru_cell_506069_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
!
²
while_body_506566
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_1_506588_0:	-
while_gru_cell_1_506590_0:
-
while_gru_cell_1_506592_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_1_506588:	+
while_gru_cell_1_506590:
+
while_gru_cell_1_506592:
¢(while/gru_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_506588_0while_gru_cell_1_506590_0while_gru_cell_1_506592_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_506513r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:01while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw

while/NoOpNoOp)^while/gru_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_1_506588while_gru_cell_1_506588_0"4
while_gru_cell_1_506590while_gru_cell_1_506590_0"4
while_gru_cell_1_506592while_gru_cell_1_506592_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ü
ª
while_cond_507330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_507330___redundant_placeholder04
0while_while_cond_507330___redundant_placeholder14
0while_while_cond_507330___redundant_placeholder24
0while_while_cond_507330___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ü
ª
while_cond_506870
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_506870___redundant_placeholder04
0while_while_cond_506870___redundant_placeholder14
0while_while_cond_506870___redundant_placeholder24
0while_while_cond_506870___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ü4

A__inference_gru_1_layer_call_and_return_conditional_losses_506447

inputs$
gru_cell_1_506369:	%
gru_cell_1_506371:
%
gru_cell_1_506373:

identity¢"gru_cell_1/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_506369gru_cell_1_506371gru_cell_1_506373*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_506368n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ú
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_506369gru_cell_1_506371gru_cell_1_506373*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_506382*
condR
while_cond_506381*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp#^gru_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£;
ï
while_body_507331
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	E
1while_gru_cell_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	C
/while_gru_cell_matmul_1_readvariableop_resource:
¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


ó
A__inference_dense_layer_call_and_return_conditional_losses_509694

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

#arbs_bot_v1_gru_1_while_cond_505849@
<arbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_loop_counterF
Barbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_maximum_iterations'
#arbs_bot_v1_gru_1_while_placeholder)
%arbs_bot_v1_gru_1_while_placeholder_1)
%arbs_bot_v1_gru_1_while_placeholder_2B
>arbs_bot_v1_gru_1_while_less_arbs_bot_v1_gru_1_strided_slice_1X
Tarbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_cond_505849___redundant_placeholder0X
Tarbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_cond_505849___redundant_placeholder1X
Tarbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_cond_505849___redundant_placeholder2X
Tarbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_cond_505849___redundant_placeholder3$
 arbs_bot_v1_gru_1_while_identity
ª
arbs_bot_v1/gru_1/while/LessLess#arbs_bot_v1_gru_1_while_placeholder>arbs_bot_v1_gru_1_while_less_arbs_bot_v1_gru_1_strided_slice_1*
T0*
_output_shapes
: o
 arbs_bot_v1/gru_1/while/IdentityIdentity arbs_bot_v1/gru_1/while/Less:z:0*
T0
*
_output_shapes
: "M
 arbs_bot_v1_gru_1_while_identity)arbs_bot_v1/gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
T
´
__inference__traced_save_510094
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop6
2savev2_gru_1_gru_cell_1_kernel_read_readvariableop@
<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop4
0savev2_gru_1_gru_cell_1_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop9
5savev2_adam_m_gru_gru_cell_kernel_read_readvariableop9
5savev2_adam_v_gru_gru_cell_kernel_read_readvariableopC
?savev2_adam_m_gru_gru_cell_recurrent_kernel_read_readvariableopC
?savev2_adam_v_gru_gru_cell_recurrent_kernel_read_readvariableop7
3savev2_adam_m_gru_gru_cell_bias_read_readvariableop7
3savev2_adam_v_gru_gru_cell_bias_read_readvariableop=
9savev2_adam_m_gru_1_gru_cell_1_kernel_read_readvariableop=
9savev2_adam_v_gru_1_gru_cell_1_kernel_read_readvariableopG
Csavev2_adam_m_gru_1_gru_cell_1_recurrent_kernel_read_readvariableopG
Csavev2_adam_v_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop;
7savev2_adam_m_gru_1_gru_cell_1_bias_read_readvariableop;
7savev2_adam_v_gru_1_gru_cell_1_bias_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableop4
0savev2_adam_m_dense_1_kernel_read_readvariableop4
0savev2_adam_v_dense_1_kernel_read_readvariableop2
.savev2_adam_m_dense_1_bias_read_readvariableop2
.savev2_adam_v_dense_1_bias_read_readvariableop4
0savev2_adam_m_dense_2_kernel_read_readvariableop4
0savev2_adam_v_dense_2_kernel_read_readvariableop2
.savev2_adam_m_dense_2_bias_read_readvariableop2
.savev2_adam_v_dense_2_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ú
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*
valueùBö+B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÃ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¢
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop2savev2_gru_1_gru_cell_1_kernel_read_readvariableop<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop0savev2_gru_1_gru_cell_1_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop5savev2_adam_m_gru_gru_cell_kernel_read_readvariableop5savev2_adam_v_gru_gru_cell_kernel_read_readvariableop?savev2_adam_m_gru_gru_cell_recurrent_kernel_read_readvariableop?savev2_adam_v_gru_gru_cell_recurrent_kernel_read_readvariableop3savev2_adam_m_gru_gru_cell_bias_read_readvariableop3savev2_adam_v_gru_gru_cell_bias_read_readvariableop9savev2_adam_m_gru_1_gru_cell_1_kernel_read_readvariableop9savev2_adam_v_gru_1_gru_cell_1_kernel_read_readvariableopCsavev2_adam_m_gru_1_gru_cell_1_recurrent_kernel_read_readvariableopCsavev2_adam_v_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop7savev2_adam_m_gru_1_gru_cell_1_bias_read_readvariableop7savev2_adam_v_gru_1_gru_cell_1_bias_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop0savev2_adam_m_dense_1_kernel_read_readvariableop0savev2_adam_v_dense_1_kernel_read_readvariableop.savev2_adam_m_dense_1_bias_read_readvariableop.savev2_adam_v_dense_1_bias_read_readvariableop0savev2_adam_m_dense_2_kernel_read_readvariableop0savev2_adam_v_dense_2_kernel_read_readvariableop.savev2_adam_m_dense_2_bias_read_readvariableop.savev2_adam_v_dense_2_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *9
dtypes/
-2+	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesõ
ò: :	 : : ::::	:
:	:
:
:	: : :	:	:
:
:	:	:
:
:
:
:	:	:	 :	 : : : : ::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	 : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	:&"
 
_output_shapes
:
:%	!

_output_shapes
:	:&
"
 
_output_shapes
:
:&"
 
_output_shapes
:
:%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:%!

_output_shapes
:	:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:%!

_output_shapes
:	:%!

_output_shapes
:	:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:%!

_output_shapes
:	:%!

_output_shapes
:	:%!

_output_shapes
:	 :%!

_output_shapes
:	 : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: :$  

_output_shapes

: : !

_output_shapes
:: "

_output_shapes
::$# 

_output_shapes

::$$ 

_output_shapes

:: %

_output_shapes
:: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: 
N

A__inference_gru_1_layer_call_and_return_conditional_losses_509209
inputs_05
"gru_cell_1_readvariableop_resource:	=
)gru_cell_1_matmul_readvariableop_resource:
?
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_509119*
condR
while_cond_509118*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0

»
&__inference_gru_1_layer_call_fn_509032
inputs_0
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_506631p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
ý
¹
&__inference_gru_1_layer_call_fn_509043

inputs
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_506961p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
!
²
while_body_506382
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_1_506404_0:	-
while_gru_cell_1_506406_0:
-
while_gru_cell_1_506408_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_1_506404:	+
while_gru_cell_1_506406:
+
while_gru_cell_1_506408:
¢(while/gru_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_506404_0while_gru_cell_1_506406_0while_gru_cell_1_506408_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_506368r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:01while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw

while/NoOpNoOp)^while/gru_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_1_506404while_gru_cell_1_506404_0"4
while_gru_cell_1_506406while_gru_cell_1_506406_0"4
while_gru_cell_1_506408while_gru_cell_1_506408_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
íM

A__inference_gru_1_layer_call_and_return_conditional_losses_509519

inputs5
"gru_cell_1_readvariableop_resource:	=
)gru_cell_1_matmul_readvariableop_resource:
?
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_509429*
condR
while_cond_509428*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í=

while_body_509274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_1_readvariableop_resource_0:	E
1while_gru_cell_1_matmul_readvariableop_resource_0:
G
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_1_readvariableop_resource:	C
/while_gru_cell_1_matmul_readvariableop_resource:
E
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ë
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¬
Ü
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_509945

inputs
states_0*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0

×
D__inference_gru_cell_layer_call_and_return_conditional_losses_506030

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¶E
·	
gru_1_while_body_507920(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0C
0gru_1_while_gru_cell_1_readvariableop_resource_0:	K
7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0:
M
9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0:

gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorA
.gru_1_while_gru_cell_1_readvariableop_resource:	I
5gru_1_while_gru_cell_1_matmul_readvariableop_resource:
K
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource:
¢,gru_1/while/gru_cell_1/MatMul/ReadVariableOp¢.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp¢%gru_1/while/gru_cell_1/ReadVariableOp
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Å
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru_1/while/gru_cell_1/unstackUnpack-gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num¦
,gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0È
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
gru_1/while/gru_cell_1/BiasAddBiasAdd'gru_1/while/gru_cell_1/MatMul:product:0'gru_1/while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
&gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_1/while/gru_cell_1/splitSplit/gru_1/while/gru_cell_1/split/split_dim:output:0'gru_1/while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitª
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¯
gru_1/while/gru_cell_1/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
 gru_1/while/gru_cell_1/BiasAdd_1BiasAdd)gru_1/while/gru_cell_1/MatMul_1:product:0'gru_1/while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
gru_1/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿs
(gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¥
gru_1/while/gru_cell_1/split_1SplitV)gru_1/while/gru_cell_1/BiasAdd_1:output:0%gru_1/while/gru_cell_1/Const:output:01gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¦
gru_1/while/gru_cell_1/addAddV2%gru_1/while/gru_cell_1/split:output:0'gru_1/while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
gru_1/while/gru_cell_1/add_1AddV2%gru_1/while/gru_cell_1/split:output:1'gru_1/while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
gru_1/while/gru_cell_1/mulMul$gru_1/while/gru_cell_1/Sigmoid_1:y:0'gru_1/while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/while/gru_cell_1/add_2AddV2%gru_1/while/gru_cell_1/split:output:2gru_1/while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru_1/while/gru_cell_1/ReluRelu gru_1/while/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/while/gru_cell_1/mul_1Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
gru_1/while/gru_cell_1/mul_2Mulgru_1/while/gru_cell_1/sub:z:0)gru_1/while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_1:z:0 gru_1/while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
6gru_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1?gru_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0 gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0^gru_1/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
gru_1/while/NoOpNoOp-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_1_matmul_readvariableop_resource7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"b
.gru_1_while_gru_cell_1_readvariableop_resource0gru_1_while_gru_cell_1_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"À
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2\
,gru_1/while/gru_cell_1/MatMul/ReadVariableOp,gru_1/while/gru_cell_1/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp2N
%gru_1/while/gru_cell_1/ReadVariableOp%gru_1/while/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
È
ö
gru_while_cond_508093$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1<
8gru_while_gru_while_cond_508093___redundant_placeholder0<
8gru_while_gru_while_cond_508093___redundant_placeholder1<
8gru_while_gru_while_cond_508093___redundant_placeholder2<
8gru_while_gru_while_cond_508093___redundant_placeholder3
gru_while_identity
r
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
í=

while_body_506871
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_1_readvariableop_resource_0:	E
1while_gru_cell_1_matmul_readvariableop_resource_0:
G
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_1_readvariableop_resource:	C
/while_gru_cell_1_matmul_readvariableop_resource:
E
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ë
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
þ
¶
$__inference_gru_layer_call_fn_508398

inputs
unknown:	
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_507420t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°O
Ï
!arbs_bot_v1_gru_while_body_505700<
8arbs_bot_v1_gru_while_arbs_bot_v1_gru_while_loop_counterB
>arbs_bot_v1_gru_while_arbs_bot_v1_gru_while_maximum_iterations%
!arbs_bot_v1_gru_while_placeholder'
#arbs_bot_v1_gru_while_placeholder_1'
#arbs_bot_v1_gru_while_placeholder_2;
7arbs_bot_v1_gru_while_arbs_bot_v1_gru_strided_slice_1_0w
sarbs_bot_v1_gru_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_tensorarrayunstack_tensorlistfromtensor_0K
8arbs_bot_v1_gru_while_gru_cell_readvariableop_resource_0:	R
?arbs_bot_v1_gru_while_gru_cell_matmul_readvariableop_resource_0:	U
Aarbs_bot_v1_gru_while_gru_cell_matmul_1_readvariableop_resource_0:
"
arbs_bot_v1_gru_while_identity$
 arbs_bot_v1_gru_while_identity_1$
 arbs_bot_v1_gru_while_identity_2$
 arbs_bot_v1_gru_while_identity_3$
 arbs_bot_v1_gru_while_identity_49
5arbs_bot_v1_gru_while_arbs_bot_v1_gru_strided_slice_1u
qarbs_bot_v1_gru_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_tensorarrayunstack_tensorlistfromtensorI
6arbs_bot_v1_gru_while_gru_cell_readvariableop_resource:	P
=arbs_bot_v1_gru_while_gru_cell_matmul_readvariableop_resource:	S
?arbs_bot_v1_gru_while_gru_cell_matmul_1_readvariableop_resource:
¢4arbs_bot_v1/gru/while/gru_cell/MatMul/ReadVariableOp¢6arbs_bot_v1/gru/while/gru_cell/MatMul_1/ReadVariableOp¢-arbs_bot_v1/gru/while/gru_cell/ReadVariableOp
Garbs_bot_v1/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ö
9arbs_bot_v1/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsarbs_bot_v1_gru_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_tensorarrayunstack_tensorlistfromtensor_0!arbs_bot_v1_gru_while_placeholderParbs_bot_v1/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0§
-arbs_bot_v1/gru/while/gru_cell/ReadVariableOpReadVariableOp8arbs_bot_v1_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
&arbs_bot_v1/gru/while/gru_cell/unstackUnpack5arbs_bot_v1/gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numµ
4arbs_bot_v1/gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp?arbs_bot_v1_gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0â
%arbs_bot_v1/gru/while/gru_cell/MatMulMatMul@arbs_bot_v1/gru/while/TensorArrayV2Read/TensorListGetItem:item:0<arbs_bot_v1/gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
&arbs_bot_v1/gru/while/gru_cell/BiasAddBiasAdd/arbs_bot_v1/gru/while/gru_cell/MatMul:product:0/arbs_bot_v1/gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.arbs_bot_v1/gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$arbs_bot_v1/gru/while/gru_cell/splitSplit7arbs_bot_v1/gru/while/gru_cell/split/split_dim:output:0/arbs_bot_v1/gru/while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitº
6arbs_bot_v1/gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpAarbs_bot_v1_gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0É
'arbs_bot_v1/gru/while/gru_cell/MatMul_1MatMul#arbs_bot_v1_gru_while_placeholder_2>arbs_bot_v1/gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
(arbs_bot_v1/gru/while/gru_cell/BiasAdd_1BiasAdd1arbs_bot_v1/gru/while/gru_cell/MatMul_1:product:0/arbs_bot_v1/gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$arbs_bot_v1/gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ{
0arbs_bot_v1/gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÅ
&arbs_bot_v1/gru/while/gru_cell/split_1SplitV1arbs_bot_v1/gru/while/gru_cell/BiasAdd_1:output:0-arbs_bot_v1/gru/while/gru_cell/Const:output:09arbs_bot_v1/gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¾
"arbs_bot_v1/gru/while/gru_cell/addAddV2-arbs_bot_v1/gru/while/gru_cell/split:output:0/arbs_bot_v1/gru/while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&arbs_bot_v1/gru/while/gru_cell/SigmoidSigmoid&arbs_bot_v1/gru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
$arbs_bot_v1/gru/while/gru_cell/add_1AddV2-arbs_bot_v1/gru/while/gru_cell/split:output:1/arbs_bot_v1/gru/while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(arbs_bot_v1/gru/while/gru_cell/Sigmoid_1Sigmoid(arbs_bot_v1/gru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
"arbs_bot_v1/gru/while/gru_cell/mulMul,arbs_bot_v1/gru/while/gru_cell/Sigmoid_1:y:0/arbs_bot_v1/gru/while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
$arbs_bot_v1/gru/while/gru_cell/add_2AddV2-arbs_bot_v1/gru/while/gru_cell/split:output:2&arbs_bot_v1/gru/while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#arbs_bot_v1/gru/while/gru_cell/ReluRelu(arbs_bot_v1/gru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
$arbs_bot_v1/gru/while/gru_cell/mul_1Mul*arbs_bot_v1/gru/while/gru_cell/Sigmoid:y:0#arbs_bot_v1_gru_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$arbs_bot_v1/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?·
"arbs_bot_v1/gru/while/gru_cell/subSub-arbs_bot_v1/gru/while/gru_cell/sub/x:output:0*arbs_bot_v1/gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
$arbs_bot_v1/gru/while/gru_cell/mul_2Mul&arbs_bot_v1/gru/while/gru_cell/sub:z:01arbs_bot_v1/gru/while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
$arbs_bot_v1/gru/while/gru_cell/add_3AddV2(arbs_bot_v1/gru/while/gru_cell/mul_1:z:0(arbs_bot_v1/gru/while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:arbs_bot_v1/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#arbs_bot_v1_gru_while_placeholder_1!arbs_bot_v1_gru_while_placeholder(arbs_bot_v1/gru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ]
arbs_bot_v1/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
arbs_bot_v1/gru/while/addAddV2!arbs_bot_v1_gru_while_placeholder$arbs_bot_v1/gru/while/add/y:output:0*
T0*
_output_shapes
: _
arbs_bot_v1/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :§
arbs_bot_v1/gru/while/add_1AddV28arbs_bot_v1_gru_while_arbs_bot_v1_gru_while_loop_counter&arbs_bot_v1/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 
arbs_bot_v1/gru/while/IdentityIdentityarbs_bot_v1/gru/while/add_1:z:0^arbs_bot_v1/gru/while/NoOp*
T0*
_output_shapes
: ª
 arbs_bot_v1/gru/while/Identity_1Identity>arbs_bot_v1_gru_while_arbs_bot_v1_gru_while_maximum_iterations^arbs_bot_v1/gru/while/NoOp*
T0*
_output_shapes
: 
 arbs_bot_v1/gru/while/Identity_2Identityarbs_bot_v1/gru/while/add:z:0^arbs_bot_v1/gru/while/NoOp*
T0*
_output_shapes
: ¶
 arbs_bot_v1/gru/while/Identity_3IdentityJarbs_bot_v1/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^arbs_bot_v1/gru/while/NoOp*
T0*
_output_shapes
: ¦
 arbs_bot_v1/gru/while/Identity_4Identity(arbs_bot_v1/gru/while/gru_cell/add_3:z:0^arbs_bot_v1/gru/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
arbs_bot_v1/gru/while/NoOpNoOp5^arbs_bot_v1/gru/while/gru_cell/MatMul/ReadVariableOp7^arbs_bot_v1/gru/while/gru_cell/MatMul_1/ReadVariableOp.^arbs_bot_v1/gru/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "p
5arbs_bot_v1_gru_while_arbs_bot_v1_gru_strided_slice_17arbs_bot_v1_gru_while_arbs_bot_v1_gru_strided_slice_1_0"
?arbs_bot_v1_gru_while_gru_cell_matmul_1_readvariableop_resourceAarbs_bot_v1_gru_while_gru_cell_matmul_1_readvariableop_resource_0"
=arbs_bot_v1_gru_while_gru_cell_matmul_readvariableop_resource?arbs_bot_v1_gru_while_gru_cell_matmul_readvariableop_resource_0"r
6arbs_bot_v1_gru_while_gru_cell_readvariableop_resource8arbs_bot_v1_gru_while_gru_cell_readvariableop_resource_0"I
arbs_bot_v1_gru_while_identity'arbs_bot_v1/gru/while/Identity:output:0"M
 arbs_bot_v1_gru_while_identity_1)arbs_bot_v1/gru/while/Identity_1:output:0"M
 arbs_bot_v1_gru_while_identity_2)arbs_bot_v1/gru/while/Identity_2:output:0"M
 arbs_bot_v1_gru_while_identity_3)arbs_bot_v1/gru/while/Identity_3:output:0"M
 arbs_bot_v1_gru_while_identity_4)arbs_bot_v1/gru/while/Identity_4:output:0"è
qarbs_bot_v1_gru_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_tensorarrayunstack_tensorlistfromtensorsarbs_bot_v1_gru_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2l
4arbs_bot_v1/gru/while/gru_cell/MatMul/ReadVariableOp4arbs_bot_v1/gru/while/gru_cell/MatMul/ReadVariableOp2p
6arbs_bot_v1/gru/while/gru_cell/MatMul_1/ReadVariableOp6arbs_bot_v1/gru/while/gru_cell/MatMul_1/ReadVariableOp2^
-arbs_bot_v1/gru/while/gru_cell/ReadVariableOp-arbs_bot_v1/gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ü
ª
while_cond_506381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_506381___redundant_placeholder04
0while_while_cond_506381___redundant_placeholder14
0while_while_cond_506381___redundant_placeholder24
0while_while_cond_506381___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ä

G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507493

inputs

gru_507463:	

gru_507465:	

gru_507467:

gru_1_507470:	 
gru_1_507472:
 
gru_1_507474:

dense_507477:	 
dense_507479:  
dense_1_507482: 
dense_1_507484: 
dense_2_507487:
dense_2_507489:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCallï
gru/StatefulPartitionedCallStatefulPartitionedCallinputs
gru_507463
gru_507465
gru_507467*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_507420
gru_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0gru_1_507470gru_1_507472gru_1_507474*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_507245
dense/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_507477dense_507479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506980
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_507482dense_1_507484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_506997
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_507487dense_2_507489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_507013w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
ª
while_cond_506224
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_506224___redundant_placeholder04
0while_while_cond_506224___redundant_placeholder14
0while_while_cond_506224___redundant_placeholder24
0while_while_cond_506224___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

¾
,__inference_arbs_bot_v1_layer_call_fn_507677

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507020o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
ª
while_cond_509428
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_509428___redundant_placeholder04
0while_while_cond_509428___redundant_placeholder14
0while_while_cond_509428___redundant_placeholder24
0while_while_cond_509428___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ü
ª
while_cond_506042
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_506042___redundant_placeholder04
0while_while_cond_506042___redundant_placeholder14
0while_while_cond_506042___redundant_placeholder24
0while_while_cond_506042___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

¿
,__inference_arbs_bot_v1_layer_call_fn_507047
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507020o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
í=

while_body_507155
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_1_readvariableop_resource_0:	E
1while_gru_cell_1_matmul_readvariableop_resource_0:
G
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_1_readvariableop_resource:	C
/while_gru_cell_1_matmul_readvariableop_resource:
E
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ë
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¤
Ú
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_506513

inputs

states*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
N

A__inference_gru_1_layer_call_and_return_conditional_losses_509364
inputs_05
"gru_cell_1_readvariableop_resource:	=
)gru_cell_1_matmul_readvariableop_resource:
?
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_509274*
condR
while_cond_509273*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0


ô
C__inference_dense_1_layer_call_and_return_conditional_losses_509714

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ô
C__inference_dense_1_layer_call_and_return_conditional_losses_506997

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
þ

gru_1_while_cond_508243(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_508243___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_508243___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_508243___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_508243___redundant_placeholder3
gru_1_while_identity
z
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
£;
ï
while_body_508921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	E
1while_gru_cell_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	C
/while_gru_cell_matmul_1_readvariableop_resource:
¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
À

(__inference_dense_1_layer_call_fn_509703

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_506997o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¼
¢
while_body_506225
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_506247_0:	*
while_gru_cell_506249_0:	+
while_gru_cell_506251_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_506247:	(
while_gru_cell_506249:	)
while_gru_cell_506251:
¢&while/gru_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0÷
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_506247_0while_gru_cell_506249_0while_gru_cell_506251_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_506173Ø
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "0
while_gru_cell_506247while_gru_cell_506247_0"0
while_gru_cell_506249while_gru_cell_506249_0"0
while_gru_cell_506251while_gru_cell_506251_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
íM

A__inference_gru_1_layer_call_and_return_conditional_losses_507245

inputs5
"gru_cell_1_readvariableop_resource:	=
)gru_cell_1_matmul_readvariableop_resource:
?
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_507155*
condR
while_cond_507154*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
L
ö
?__inference_gru_layer_call_and_return_conditional_losses_508704
inputs_03
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	=
)gru_cell_matmul_1_readvariableop_resource:

identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÁ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split|
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_508615*
condR
while_cond_508614*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
á

·
$__inference_signature_wrapper_507648
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_505960o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
µ

Ù
)__inference_gru_cell_layer_call_fn_509761

inputs
states_0
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_506173p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0
L
ö
?__inference_gru_layer_call_and_return_conditional_losses_508551
inputs_03
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	=
)gru_cell_matmul_1_readvariableop_resource:

identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÁ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split|
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_508462*
condR
while_cond_508461*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
Ä

G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507020

inputs

gru_506800:	

gru_506802:	

gru_506804:

gru_1_506962:	 
gru_1_506964:
 
gru_1_506966:

dense_506981:	 
dense_506983:  
dense_1_506998: 
dense_1_507000: 
dense_2_507014:
dense_2_507016:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCallï
gru/StatefulPartitionedCallStatefulPartitionedCallinputs
gru_506800
gru_506802
gru_506804*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_506799
gru_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0gru_1_506962gru_1_506964gru_1_506966*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_506961
dense/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_506981dense_506983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506980
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_506998dense_1_507000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_506997
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_507014dense_2_507016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_507013w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ú
!arbs_bot_v1_gru_while_cond_505699<
8arbs_bot_v1_gru_while_arbs_bot_v1_gru_while_loop_counterB
>arbs_bot_v1_gru_while_arbs_bot_v1_gru_while_maximum_iterations%
!arbs_bot_v1_gru_while_placeholder'
#arbs_bot_v1_gru_while_placeholder_1'
#arbs_bot_v1_gru_while_placeholder_2>
:arbs_bot_v1_gru_while_less_arbs_bot_v1_gru_strided_slice_1T
Parbs_bot_v1_gru_while_arbs_bot_v1_gru_while_cond_505699___redundant_placeholder0T
Parbs_bot_v1_gru_while_arbs_bot_v1_gru_while_cond_505699___redundant_placeholder1T
Parbs_bot_v1_gru_while_arbs_bot_v1_gru_while_cond_505699___redundant_placeholder2T
Parbs_bot_v1_gru_while_arbs_bot_v1_gru_while_cond_505699___redundant_placeholder3"
arbs_bot_v1_gru_while_identity
¢
arbs_bot_v1/gru/while/LessLess!arbs_bot_v1_gru_while_placeholder:arbs_bot_v1_gru_while_less_arbs_bot_v1_gru_strided_slice_1*
T0*
_output_shapes
: k
arbs_bot_v1/gru/while/IdentityIdentityarbs_bot_v1/gru/while/Less:z:0*
T0
*
_output_shapes
: "I
arbs_bot_v1_gru_while_identity'arbs_bot_v1/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¨
¸
$__inference_gru_layer_call_fn_508365
inputs_0
unknown:	
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_506107}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
Ü
ª
while_cond_507154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_507154___redundant_placeholder04
0while_while_cond_507154___redundant_placeholder14
0while_while_cond_507154___redundant_placeholder24
0while_while_cond_507154___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ú¹
ô	
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_508354

inputs7
$gru_gru_cell_readvariableop_resource:	>
+gru_gru_cell_matmul_readvariableop_resource:	A
-gru_gru_cell_matmul_1_readvariableop_resource:
;
(gru_1_gru_cell_1_readvariableop_resource:	C
/gru_1_gru_cell_1_matmul_readvariableop_resource:
E
1gru_1_gru_cell_1_matmul_1_readvariableop_resource:
7
$dense_matmul_readvariableop_resource:	 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢"gru/gru_cell/MatMul/ReadVariableOp¢$gru/gru_cell/MatMul_1/ReadVariableOp¢gru/gru_cell/ReadVariableOp¢	gru/while¢&gru_1/gru_cell_1/MatMul/ReadVariableOp¢(gru_1/gru_cell_1/MatMul_1/ReadVariableOp¢gru_1/gru_cell_1/ReadVariableOp¢gru_1/while?
	gru/ShapeShapeinputs*
T0*
_output_shapes
:a
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    y
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:c
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒc
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ý
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0{
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÍ
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru/gru_cell/MatMul_1MatMulgru/zeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿi
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒJ
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿX
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *!
bodyR
gru_while_body_508094*!
condR
gru_while_cond_508093*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿe
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    N
gru_1/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
:c
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_1/transpose	Transposegru/transpose_1:y:0gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
:e
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_1/gru_cell_1/ReadVariableOpReadVariableOp(gru_1_gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_1/gru_cell_1/unstackUnpack'gru_1/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¤
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/BiasAddBiasAdd!gru_1/gru_cell_1/MatMul:product:0!gru_1/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_1/gru_cell_1/splitSplit)gru_1/gru_cell_1/split/split_dim:output:0!gru_1/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
(gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_1/gru_cell_1/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
gru_1/gru_cell_1/BiasAdd_1BiasAdd#gru_1/gru_cell_1/MatMul_1:product:0!gru_1/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
gru_1/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿm
"gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/split_1SplitV#gru_1/gru_cell_1/BiasAdd_1:output:0gru_1/gru_cell_1/Const:output:0+gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_1/gru_cell_1/addAddV2gru_1/gru_cell_1/split:output:0!gru_1/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/add_1AddV2gru_1/gru_cell_1/split:output:1!gru_1/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/mulMulgru_1/gru_cell_1/Sigmoid_1:y:0!gru_1/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/add_2AddV2gru_1/gru_cell_1/split:output:2gru_1/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
gru_1/gru_cell_1/ReluRelugru_1/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/mul_1Mulgru_1/gru_cell_1/Sigmoid:y:0gru_1/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/mul_2Mulgru_1/gru_cell_1/sub:z:0#gru_1/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_1:z:0gru_1/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   d
"gru_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :×
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0+gru_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_1_readvariableop_resource/gru_1_gru_cell_1_matmul_readvariableop_resource1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_1_while_body_508244*#
condR
gru_1_while_cond_508243*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   é
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsn
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
dense/MatMulMatMulgru_1/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while'^gru_1/gru_cell_1/MatMul/ReadVariableOp)^gru_1/gru_cell_1/MatMul_1/ReadVariableOp ^gru_1/gru_cell_1/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while2P
&gru_1/gru_cell_1/MatMul/ReadVariableOp&gru_1/gru_cell_1/MatMul/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_1/ReadVariableOp(gru_1/gru_cell_1/MatMul_1/ReadVariableOp2B
gru_1/gru_cell_1/ReadVariableOpgru_1/gru_cell_1/ReadVariableOp2
gru_1/whilegru_1/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü4

A__inference_gru_1_layer_call_and_return_conditional_losses_506631

inputs$
gru_cell_1_506553:	%
gru_cell_1_506555:
%
gru_cell_1_506557:

identity¢"gru_cell_1/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_506553gru_cell_1_506555gru_cell_1_506557*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_506513n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ú
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_506553gru_cell_1_506555gru_cell_1_506557*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_506566*
condR
while_cond_506565*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
NoOpNoOp#^gru_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿

&__inference_dense_layer_call_fn_509683

inputs
unknown:	 
	unknown_0: 
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506980o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£;
ï
while_body_508462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	E
1while_gru_cell_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	C
/while_gru_cell_matmul_1_readvariableop_resource:
¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


ó
A__inference_dense_layer_call_and_return_conditional_losses_506980

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÅK
ô
?__inference_gru_layer_call_and_return_conditional_losses_508857

inputs3
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	=
)gru_cell_matmul_1_readvariableop_resource:

identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÁ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split|
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_508768*
condR
while_cond_508767*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
ª
while_cond_508614
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_508614___redundant_placeholder04
0while_while_cond_508614___redundant_placeholder14
0while_while_cond_508614___redundant_placeholder24
0while_while_cond_508614___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ç

G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507615
input_1

gru_507585:	

gru_507587:	

gru_507589:

gru_1_507592:	 
gru_1_507594:
 
gru_1_507596:

dense_507599:	 
dense_507601:  
dense_1_507604: 
dense_1_507606: 
dense_2_507609:
dense_2_507611:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCallð
gru/StatefulPartitionedCallStatefulPartitionedCallinput_1
gru_507585
gru_507587
gru_507589*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_507420
gru_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0gru_1_507592gru_1_507594gru_1_507596*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_507245
dense/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_507599dense_507601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506980
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_507604dense_1_507606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_506997
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_507609dense_2_507611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_507013w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

¾
,__inference_arbs_bot_v1_layer_call_fn_507706

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¹
&__inference_gru_1_layer_call_fn_509054

inputs
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_507245p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
Ù
D__inference_gru_cell_layer_call_and_return_conditional_losses_509800

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0
ì3
ö
?__inference_gru_layer_call_and_return_conditional_losses_506289

inputs"
gru_cell_506213:	"
gru_cell_506215:	#
gru_cell_506217:

identity¢ gru_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¼
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_506213gru_cell_506215gru_cell_506217*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_506173n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_506213gru_cell_506215gru_cell_506217*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_506225*
condR
while_cond_506224*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
íM

A__inference_gru_1_layer_call_and_return_conditional_losses_506961

inputs5
"gru_cell_1_readvariableop_resource:	=
)gru_cell_1_matmul_readvariableop_resource:
?
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_506871*
condR
while_cond_506870*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÞÞ

!__inference__wrapped_model_505960
input_1C
0arbs_bot_v1_gru_gru_cell_readvariableop_resource:	J
7arbs_bot_v1_gru_gru_cell_matmul_readvariableop_resource:	M
9arbs_bot_v1_gru_gru_cell_matmul_1_readvariableop_resource:
G
4arbs_bot_v1_gru_1_gru_cell_1_readvariableop_resource:	O
;arbs_bot_v1_gru_1_gru_cell_1_matmul_readvariableop_resource:
Q
=arbs_bot_v1_gru_1_gru_cell_1_matmul_1_readvariableop_resource:
C
0arbs_bot_v1_dense_matmul_readvariableop_resource:	 ?
1arbs_bot_v1_dense_biasadd_readvariableop_resource: D
2arbs_bot_v1_dense_1_matmul_readvariableop_resource: A
3arbs_bot_v1_dense_1_biasadd_readvariableop_resource:D
2arbs_bot_v1_dense_2_matmul_readvariableop_resource:A
3arbs_bot_v1_dense_2_biasadd_readvariableop_resource:
identity¢(arbs_bot_v1/dense/BiasAdd/ReadVariableOp¢'arbs_bot_v1/dense/MatMul/ReadVariableOp¢*arbs_bot_v1/dense_1/BiasAdd/ReadVariableOp¢)arbs_bot_v1/dense_1/MatMul/ReadVariableOp¢*arbs_bot_v1/dense_2/BiasAdd/ReadVariableOp¢)arbs_bot_v1/dense_2/MatMul/ReadVariableOp¢.arbs_bot_v1/gru/gru_cell/MatMul/ReadVariableOp¢0arbs_bot_v1/gru/gru_cell/MatMul_1/ReadVariableOp¢'arbs_bot_v1/gru/gru_cell/ReadVariableOp¢arbs_bot_v1/gru/while¢2arbs_bot_v1/gru_1/gru_cell_1/MatMul/ReadVariableOp¢4arbs_bot_v1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp¢+arbs_bot_v1/gru_1/gru_cell_1/ReadVariableOp¢arbs_bot_v1/gru_1/whileL
arbs_bot_v1/gru/ShapeShapeinput_1*
T0*
_output_shapes
:m
#arbs_bot_v1/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%arbs_bot_v1/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%arbs_bot_v1/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
arbs_bot_v1/gru/strided_sliceStridedSlicearbs_bot_v1/gru/Shape:output:0,arbs_bot_v1/gru/strided_slice/stack:output:0.arbs_bot_v1/gru/strided_slice/stack_1:output:0.arbs_bot_v1/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
arbs_bot_v1/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :£
arbs_bot_v1/gru/zeros/packedPack&arbs_bot_v1/gru/strided_slice:output:0'arbs_bot_v1/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
arbs_bot_v1/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
arbs_bot_v1/gru/zerosFill%arbs_bot_v1/gru/zeros/packed:output:0$arbs_bot_v1/gru/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
arbs_bot_v1/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
arbs_bot_v1/gru/transpose	Transposeinput_1'arbs_bot_v1/gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
arbs_bot_v1/gru/Shape_1Shapearbs_bot_v1/gru/transpose:y:0*
T0*
_output_shapes
:o
%arbs_bot_v1/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'arbs_bot_v1/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'arbs_bot_v1/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
arbs_bot_v1/gru/strided_slice_1StridedSlice arbs_bot_v1/gru/Shape_1:output:0.arbs_bot_v1/gru/strided_slice_1/stack:output:00arbs_bot_v1/gru/strided_slice_1/stack_1:output:00arbs_bot_v1/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+arbs_bot_v1/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿä
arbs_bot_v1/gru/TensorArrayV2TensorListReserve4arbs_bot_v1/gru/TensorArrayV2/element_shape:output:0(arbs_bot_v1/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Earbs_bot_v1/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
7arbs_bot_v1/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorarbs_bot_v1/gru/transpose:y:0Narbs_bot_v1/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒo
%arbs_bot_v1/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'arbs_bot_v1/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'arbs_bot_v1/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¹
arbs_bot_v1/gru/strided_slice_2StridedSlicearbs_bot_v1/gru/transpose:y:0.arbs_bot_v1/gru/strided_slice_2/stack:output:00arbs_bot_v1/gru/strided_slice_2/stack_1:output:00arbs_bot_v1/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
'arbs_bot_v1/gru/gru_cell/ReadVariableOpReadVariableOp0arbs_bot_v1_gru_gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0
 arbs_bot_v1/gru/gru_cell/unstackUnpack/arbs_bot_v1/gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num§
.arbs_bot_v1/gru/gru_cell/MatMul/ReadVariableOpReadVariableOp7arbs_bot_v1_gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¾
arbs_bot_v1/gru/gru_cell/MatMulMatMul(arbs_bot_v1/gru/strided_slice_2:output:06arbs_bot_v1/gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 arbs_bot_v1/gru/gru_cell/BiasAddBiasAdd)arbs_bot_v1/gru/gru_cell/MatMul:product:0)arbs_bot_v1/gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
(arbs_bot_v1/gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿñ
arbs_bot_v1/gru/gru_cell/splitSplit1arbs_bot_v1/gru/gru_cell/split/split_dim:output:0)arbs_bot_v1/gru/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¬
0arbs_bot_v1/gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp9arbs_bot_v1_gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¸
!arbs_bot_v1/gru/gru_cell/MatMul_1MatMularbs_bot_v1/gru/zeros:output:08arbs_bot_v1/gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
"arbs_bot_v1/gru/gru_cell/BiasAdd_1BiasAdd+arbs_bot_v1/gru/gru_cell/MatMul_1:product:0)arbs_bot_v1/gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
arbs_bot_v1/gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿu
*arbs_bot_v1/gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ­
 arbs_bot_v1/gru/gru_cell/split_1SplitV+arbs_bot_v1/gru/gru_cell/BiasAdd_1:output:0'arbs_bot_v1/gru/gru_cell/Const:output:03arbs_bot_v1/gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¬
arbs_bot_v1/gru/gru_cell/addAddV2'arbs_bot_v1/gru/gru_cell/split:output:0)arbs_bot_v1/gru/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 arbs_bot_v1/gru/gru_cell/SigmoidSigmoid arbs_bot_v1/gru/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
arbs_bot_v1/gru/gru_cell/add_1AddV2'arbs_bot_v1/gru/gru_cell/split:output:1)arbs_bot_v1/gru/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"arbs_bot_v1/gru/gru_cell/Sigmoid_1Sigmoid"arbs_bot_v1/gru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
arbs_bot_v1/gru/gru_cell/mulMul&arbs_bot_v1/gru/gru_cell/Sigmoid_1:y:0)arbs_bot_v1/gru/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
arbs_bot_v1/gru/gru_cell/add_2AddV2'arbs_bot_v1/gru/gru_cell/split:output:2 arbs_bot_v1/gru/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
arbs_bot_v1/gru/gru_cell/ReluRelu"arbs_bot_v1/gru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
arbs_bot_v1/gru/gru_cell/mul_1Mul$arbs_bot_v1/gru/gru_cell/Sigmoid:y:0arbs_bot_v1/gru/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
arbs_bot_v1/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¥
arbs_bot_v1/gru/gru_cell/subSub'arbs_bot_v1/gru/gru_cell/sub/x:output:0$arbs_bot_v1/gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
arbs_bot_v1/gru/gru_cell/mul_2Mul arbs_bot_v1/gru/gru_cell/sub:z:0+arbs_bot_v1/gru/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
arbs_bot_v1/gru/gru_cell/add_3AddV2"arbs_bot_v1/gru/gru_cell/mul_1:z:0"arbs_bot_v1/gru/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
-arbs_bot_v1/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   è
arbs_bot_v1/gru/TensorArrayV2_1TensorListReserve6arbs_bot_v1/gru/TensorArrayV2_1/element_shape:output:0(arbs_bot_v1/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒV
arbs_bot_v1/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(arbs_bot_v1/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿd
"arbs_bot_v1/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
arbs_bot_v1/gru/whileWhile+arbs_bot_v1/gru/while/loop_counter:output:01arbs_bot_v1/gru/while/maximum_iterations:output:0arbs_bot_v1/gru/time:output:0(arbs_bot_v1/gru/TensorArrayV2_1:handle:0arbs_bot_v1/gru/zeros:output:0(arbs_bot_v1/gru/strided_slice_1:output:0Garbs_bot_v1/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:00arbs_bot_v1_gru_gru_cell_readvariableop_resource7arbs_bot_v1_gru_gru_cell_matmul_readvariableop_resource9arbs_bot_v1_gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *-
body%R#
!arbs_bot_v1_gru_while_body_505700*-
cond%R#
!arbs_bot_v1_gru_while_cond_505699*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
@arbs_bot_v1/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ó
2arbs_bot_v1/gru/TensorArrayV2Stack/TensorListStackTensorListStackarbs_bot_v1/gru/while:output:3Iarbs_bot_v1/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0x
%arbs_bot_v1/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿq
'arbs_bot_v1/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'arbs_bot_v1/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
arbs_bot_v1/gru/strided_slice_3StridedSlice;arbs_bot_v1/gru/TensorArrayV2Stack/TensorListStack:tensor:0.arbs_bot_v1/gru/strided_slice_3/stack:output:00arbs_bot_v1/gru/strided_slice_3/stack_1:output:00arbs_bot_v1/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masku
 arbs_bot_v1/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ç
arbs_bot_v1/gru/transpose_1	Transpose;arbs_bot_v1/gru/TensorArrayV2Stack/TensorListStack:tensor:0)arbs_bot_v1/gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
arbs_bot_v1/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    f
arbs_bot_v1/gru_1/ShapeShapearbs_bot_v1/gru/transpose_1:y:0*
T0*
_output_shapes
:o
%arbs_bot_v1/gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'arbs_bot_v1/gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'arbs_bot_v1/gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
arbs_bot_v1/gru_1/strided_sliceStridedSlice arbs_bot_v1/gru_1/Shape:output:0.arbs_bot_v1/gru_1/strided_slice/stack:output:00arbs_bot_v1/gru_1/strided_slice/stack_1:output:00arbs_bot_v1/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 arbs_bot_v1/gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :©
arbs_bot_v1/gru_1/zeros/packedPack(arbs_bot_v1/gru_1/strided_slice:output:0)arbs_bot_v1/gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:b
arbs_bot_v1/gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    £
arbs_bot_v1/gru_1/zerosFill'arbs_bot_v1/gru_1/zeros/packed:output:0&arbs_bot_v1/gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 arbs_bot_v1/gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
arbs_bot_v1/gru_1/transpose	Transposearbs_bot_v1/gru/transpose_1:y:0)arbs_bot_v1/gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
arbs_bot_v1/gru_1/Shape_1Shapearbs_bot_v1/gru_1/transpose:y:0*
T0*
_output_shapes
:q
'arbs_bot_v1/gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)arbs_bot_v1/gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)arbs_bot_v1/gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!arbs_bot_v1/gru_1/strided_slice_1StridedSlice"arbs_bot_v1/gru_1/Shape_1:output:00arbs_bot_v1/gru_1/strided_slice_1/stack:output:02arbs_bot_v1/gru_1/strided_slice_1/stack_1:output:02arbs_bot_v1/gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-arbs_bot_v1/gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
arbs_bot_v1/gru_1/TensorArrayV2TensorListReserve6arbs_bot_v1/gru_1/TensorArrayV2/element_shape:output:0*arbs_bot_v1/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Garbs_bot_v1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
9arbs_bot_v1/gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorarbs_bot_v1/gru_1/transpose:y:0Parbs_bot_v1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒq
'arbs_bot_v1/gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)arbs_bot_v1/gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)arbs_bot_v1/gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
!arbs_bot_v1/gru_1/strided_slice_2StridedSlicearbs_bot_v1/gru_1/transpose:y:00arbs_bot_v1/gru_1/strided_slice_2/stack:output:02arbs_bot_v1/gru_1/strided_slice_2/stack_1:output:02arbs_bot_v1/gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¡
+arbs_bot_v1/gru_1/gru_cell_1/ReadVariableOpReadVariableOp4arbs_bot_v1_gru_1_gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0
$arbs_bot_v1/gru_1/gru_cell_1/unstackUnpack3arbs_bot_v1/gru_1/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num°
2arbs_bot_v1/gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp;arbs_bot_v1_gru_1_gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0È
#arbs_bot_v1/gru_1/gru_cell_1/MatMulMatMul*arbs_bot_v1/gru_1/strided_slice_2:output:0:arbs_bot_v1/gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
$arbs_bot_v1/gru_1/gru_cell_1/BiasAddBiasAdd-arbs_bot_v1/gru_1/gru_cell_1/MatMul:product:0-arbs_bot_v1/gru_1/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,arbs_bot_v1/gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
"arbs_bot_v1/gru_1/gru_cell_1/splitSplit5arbs_bot_v1/gru_1/gru_cell_1/split/split_dim:output:0-arbs_bot_v1/gru_1/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split´
4arbs_bot_v1/gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp=arbs_bot_v1_gru_1_gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Â
%arbs_bot_v1/gru_1/gru_cell_1/MatMul_1MatMul arbs_bot_v1/gru_1/zeros:output:0<arbs_bot_v1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
&arbs_bot_v1/gru_1/gru_cell_1/BiasAdd_1BiasAdd/arbs_bot_v1/gru_1/gru_cell_1/MatMul_1:product:0-arbs_bot_v1/gru_1/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"arbs_bot_v1/gru_1/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿy
.arbs_bot_v1/gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ½
$arbs_bot_v1/gru_1/gru_cell_1/split_1SplitV/arbs_bot_v1/gru_1/gru_cell_1/BiasAdd_1:output:0+arbs_bot_v1/gru_1/gru_cell_1/Const:output:07arbs_bot_v1/gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¸
 arbs_bot_v1/gru_1/gru_cell_1/addAddV2+arbs_bot_v1/gru_1/gru_cell_1/split:output:0-arbs_bot_v1/gru_1/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$arbs_bot_v1/gru_1/gru_cell_1/SigmoidSigmoid$arbs_bot_v1/gru_1/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"arbs_bot_v1/gru_1/gru_cell_1/add_1AddV2+arbs_bot_v1/gru_1/gru_cell_1/split:output:1-arbs_bot_v1/gru_1/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&arbs_bot_v1/gru_1/gru_cell_1/Sigmoid_1Sigmoid&arbs_bot_v1/gru_1/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
 arbs_bot_v1/gru_1/gru_cell_1/mulMul*arbs_bot_v1/gru_1/gru_cell_1/Sigmoid_1:y:0-arbs_bot_v1/gru_1/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
"arbs_bot_v1/gru_1/gru_cell_1/add_2AddV2+arbs_bot_v1/gru_1/gru_cell_1/split:output:2$arbs_bot_v1/gru_1/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!arbs_bot_v1/gru_1/gru_cell_1/ReluRelu&arbs_bot_v1/gru_1/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"arbs_bot_v1/gru_1/gru_cell_1/mul_1Mul(arbs_bot_v1/gru_1/gru_cell_1/Sigmoid:y:0 arbs_bot_v1/gru_1/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"arbs_bot_v1/gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?±
 arbs_bot_v1/gru_1/gru_cell_1/subSub+arbs_bot_v1/gru_1/gru_cell_1/sub/x:output:0(arbs_bot_v1/gru_1/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
"arbs_bot_v1/gru_1/gru_cell_1/mul_2Mul$arbs_bot_v1/gru_1/gru_cell_1/sub:z:0/arbs_bot_v1/gru_1/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
"arbs_bot_v1/gru_1/gru_cell_1/add_3AddV2&arbs_bot_v1/gru_1/gru_cell_1/mul_1:z:0&arbs_bot_v1/gru_1/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/arbs_bot_v1/gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   p
.arbs_bot_v1/gru_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :û
!arbs_bot_v1/gru_1/TensorArrayV2_1TensorListReserve8arbs_bot_v1/gru_1/TensorArrayV2_1/element_shape:output:07arbs_bot_v1/gru_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
arbs_bot_v1/gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : u
*arbs_bot_v1/gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿf
$arbs_bot_v1/gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : §
arbs_bot_v1/gru_1/whileWhile-arbs_bot_v1/gru_1/while/loop_counter:output:03arbs_bot_v1/gru_1/while/maximum_iterations:output:0arbs_bot_v1/gru_1/time:output:0*arbs_bot_v1/gru_1/TensorArrayV2_1:handle:0 arbs_bot_v1/gru_1/zeros:output:0*arbs_bot_v1/gru_1/strided_slice_1:output:0Iarbs_bot_v1/gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:04arbs_bot_v1_gru_1_gru_cell_1_readvariableop_resource;arbs_bot_v1_gru_1_gru_cell_1_matmul_readvariableop_resource=arbs_bot_v1_gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( */
body'R%
#arbs_bot_v1_gru_1_while_body_505850*/
cond'R%
#arbs_bot_v1_gru_1_while_cond_505849*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Barbs_bot_v1/gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
4arbs_bot_v1/gru_1/TensorArrayV2Stack/TensorListStackTensorListStack arbs_bot_v1/gru_1/while:output:3Karbs_bot_v1/gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsz
'arbs_bot_v1/gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿs
)arbs_bot_v1/gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)arbs_bot_v1/gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
!arbs_bot_v1/gru_1/strided_slice_3StridedSlice=arbs_bot_v1/gru_1/TensorArrayV2Stack/TensorListStack:tensor:00arbs_bot_v1/gru_1/strided_slice_3/stack:output:02arbs_bot_v1/gru_1/strided_slice_3/stack_1:output:02arbs_bot_v1/gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskw
"arbs_bot_v1/gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Í
arbs_bot_v1/gru_1/transpose_1	Transpose=arbs_bot_v1/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0+arbs_bot_v1/gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
arbs_bot_v1/gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
'arbs_bot_v1/dense/MatMul/ReadVariableOpReadVariableOp0arbs_bot_v1_dense_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0±
arbs_bot_v1/dense/MatMulMatMul*arbs_bot_v1/gru_1/strided_slice_3:output:0/arbs_bot_v1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(arbs_bot_v1/dense/BiasAdd/ReadVariableOpReadVariableOp1arbs_bot_v1_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¬
arbs_bot_v1/dense/BiasAddBiasAdd"arbs_bot_v1/dense/MatMul:product:00arbs_bot_v1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
arbs_bot_v1/dense/ReluRelu"arbs_bot_v1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)arbs_bot_v1/dense_1/MatMul/ReadVariableOpReadVariableOp2arbs_bot_v1_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0¯
arbs_bot_v1/dense_1/MatMulMatMul$arbs_bot_v1/dense/Relu:activations:01arbs_bot_v1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*arbs_bot_v1/dense_1/BiasAdd/ReadVariableOpReadVariableOp3arbs_bot_v1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
arbs_bot_v1/dense_1/BiasAddBiasAdd$arbs_bot_v1/dense_1/MatMul:product:02arbs_bot_v1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
arbs_bot_v1/dense_1/ReluRelu$arbs_bot_v1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)arbs_bot_v1/dense_2/MatMul/ReadVariableOpReadVariableOp2arbs_bot_v1_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0±
arbs_bot_v1/dense_2/MatMulMatMul&arbs_bot_v1/dense_1/Relu:activations:01arbs_bot_v1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*arbs_bot_v1/dense_2/BiasAdd/ReadVariableOpReadVariableOp3arbs_bot_v1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
arbs_bot_v1/dense_2/BiasAddBiasAdd$arbs_bot_v1/dense_2/MatMul:product:02arbs_bot_v1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
IdentityIdentity$arbs_bot_v1/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp)^arbs_bot_v1/dense/BiasAdd/ReadVariableOp(^arbs_bot_v1/dense/MatMul/ReadVariableOp+^arbs_bot_v1/dense_1/BiasAdd/ReadVariableOp*^arbs_bot_v1/dense_1/MatMul/ReadVariableOp+^arbs_bot_v1/dense_2/BiasAdd/ReadVariableOp*^arbs_bot_v1/dense_2/MatMul/ReadVariableOp/^arbs_bot_v1/gru/gru_cell/MatMul/ReadVariableOp1^arbs_bot_v1/gru/gru_cell/MatMul_1/ReadVariableOp(^arbs_bot_v1/gru/gru_cell/ReadVariableOp^arbs_bot_v1/gru/while3^arbs_bot_v1/gru_1/gru_cell_1/MatMul/ReadVariableOp5^arbs_bot_v1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp,^arbs_bot_v1/gru_1/gru_cell_1/ReadVariableOp^arbs_bot_v1/gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2T
(arbs_bot_v1/dense/BiasAdd/ReadVariableOp(arbs_bot_v1/dense/BiasAdd/ReadVariableOp2R
'arbs_bot_v1/dense/MatMul/ReadVariableOp'arbs_bot_v1/dense/MatMul/ReadVariableOp2X
*arbs_bot_v1/dense_1/BiasAdd/ReadVariableOp*arbs_bot_v1/dense_1/BiasAdd/ReadVariableOp2V
)arbs_bot_v1/dense_1/MatMul/ReadVariableOp)arbs_bot_v1/dense_1/MatMul/ReadVariableOp2X
*arbs_bot_v1/dense_2/BiasAdd/ReadVariableOp*arbs_bot_v1/dense_2/BiasAdd/ReadVariableOp2V
)arbs_bot_v1/dense_2/MatMul/ReadVariableOp)arbs_bot_v1/dense_2/MatMul/ReadVariableOp2`
.arbs_bot_v1/gru/gru_cell/MatMul/ReadVariableOp.arbs_bot_v1/gru/gru_cell/MatMul/ReadVariableOp2d
0arbs_bot_v1/gru/gru_cell/MatMul_1/ReadVariableOp0arbs_bot_v1/gru/gru_cell/MatMul_1/ReadVariableOp2R
'arbs_bot_v1/gru/gru_cell/ReadVariableOp'arbs_bot_v1/gru/gru_cell/ReadVariableOp2.
arbs_bot_v1/gru/whilearbs_bot_v1/gru/while2h
2arbs_bot_v1/gru_1/gru_cell_1/MatMul/ReadVariableOp2arbs_bot_v1/gru_1/gru_cell_1/MatMul/ReadVariableOp2l
4arbs_bot_v1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp4arbs_bot_v1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp2Z
+arbs_bot_v1/gru_1/gru_cell_1/ReadVariableOp+arbs_bot_v1/gru_1/gru_cell_1/ReadVariableOp22
arbs_bot_v1/gru_1/whilearbs_bot_v1/gru_1/while:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
µ

Ù
)__inference_gru_cell_layer_call_fn_509747

inputs
states_0
unknown:	
	unknown_0:	
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_506030p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0
ÅK
ô
?__inference_gru_layer_call_and_return_conditional_losses_507420

inputs3
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	=
)gru_cell_matmul_1_readvariableop_resource:

identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÁ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split|
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_507331*
condR
while_cond_507330*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
ª
while_cond_509583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_509583___redundant_placeholder04
0while_while_cond_509583___redundant_placeholder14
0while_while_cond_509583___redundant_placeholder24
0while_while_cond_509583___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¦@
ç
gru_while_body_508094$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	F
3gru_while_gru_cell_matmul_readvariableop_resource_0:	I
5gru_while_gru_cell_matmul_1_readvariableop_resource_0:

gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	D
1gru_while_gru_cell_matmul_readvariableop_resource:	G
3gru_while_gru_cell_matmul_1_readvariableop_resource:
¢(gru/while/gru_cell/MatMul/ReadVariableOp¢*gru/while/gru_cell/MatMul_1/ReadVariableOp¢!gru/while/gru_cell/ReadVariableOp
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   º
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¾
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿß
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¢
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¥
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿo
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒQ
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: 
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: 
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0^gru/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
gru/while/NoOpNoOp)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"¸
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
£;
ï
while_body_508768
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	E
1while_gru_cell_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	C
/while_gru_cell_matmul_1_readvariableop_resource:
¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¦
Ù
D__inference_gru_cell_layer_call_and_return_conditional_losses_509839

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0
¶E
·	
gru_1_while_body_508244(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0C
0gru_1_while_gru_cell_1_readvariableop_resource_0:	K
7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0:
M
9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0:

gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorA
.gru_1_while_gru_cell_1_readvariableop_resource:	I
5gru_1_while_gru_cell_1_matmul_readvariableop_resource:
K
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource:
¢,gru_1/while/gru_cell_1/MatMul/ReadVariableOp¢.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp¢%gru_1/while/gru_cell_1/ReadVariableOp
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Å
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru_1/while/gru_cell_1/unstackUnpack-gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num¦
,gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0È
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
gru_1/while/gru_cell_1/BiasAddBiasAdd'gru_1/while/gru_cell_1/MatMul:product:0'gru_1/while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
&gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_1/while/gru_cell_1/splitSplit/gru_1/while/gru_cell_1/split/split_dim:output:0'gru_1/while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitª
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¯
gru_1/while/gru_cell_1/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
 gru_1/while/gru_cell_1/BiasAdd_1BiasAdd)gru_1/while/gru_cell_1/MatMul_1:product:0'gru_1/while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
gru_1/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿs
(gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¥
gru_1/while/gru_cell_1/split_1SplitV)gru_1/while/gru_cell_1/BiasAdd_1:output:0%gru_1/while/gru_cell_1/Const:output:01gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¦
gru_1/while/gru_cell_1/addAddV2%gru_1/while/gru_cell_1/split:output:0'gru_1/while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
gru_1/while/gru_cell_1/add_1AddV2%gru_1/while/gru_cell_1/split:output:1'gru_1/while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
gru_1/while/gru_cell_1/mulMul$gru_1/while/gru_cell_1/Sigmoid_1:y:0'gru_1/while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/while/gru_cell_1/add_2AddV2%gru_1/while/gru_cell_1/split:output:2gru_1/while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru_1/while/gru_cell_1/ReluRelu gru_1/while/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/while/gru_cell_1/mul_1Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
gru_1/while/gru_cell_1/mul_2Mulgru_1/while/gru_cell_1/sub:z:0)gru_1/while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_1:z:0 gru_1/while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
6gru_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1?gru_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0 gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: 
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0^gru_1/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
gru_1/while/NoOpNoOp-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_1_matmul_readvariableop_resource7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"b
.gru_1_while_gru_cell_1_readvariableop_resource0gru_1_while_gru_cell_1_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"À
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2\
,gru_1/while/gru_cell_1/MatMul/ReadVariableOp,gru_1/while/gru_cell_1/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp2N
%gru_1/while/gru_cell_1/ReadVariableOp%gru_1/while/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¬
Ü
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_509906

inputs
states_0*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0
Ü
ª
while_cond_508920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_508920___redundant_placeholder04
0while_while_cond_508920___redundant_placeholder14
0while_while_cond_508920___redundant_placeholder24
0while_while_cond_508920___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
£;
ï
while_body_506710
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	E
1while_gru_cell_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	C
/while_gru_cell_matmul_1_readvariableop_resource:
¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ËT

#arbs_bot_v1_gru_1_while_body_505850@
<arbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_loop_counterF
Barbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_maximum_iterations'
#arbs_bot_v1_gru_1_while_placeholder)
%arbs_bot_v1_gru_1_while_placeholder_1)
%arbs_bot_v1_gru_1_while_placeholder_2?
;arbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_strided_slice_1_0{
warbs_bot_v1_gru_1_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_1_tensorarrayunstack_tensorlistfromtensor_0O
<arbs_bot_v1_gru_1_while_gru_cell_1_readvariableop_resource_0:	W
Carbs_bot_v1_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0:
Y
Earbs_bot_v1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0:
$
 arbs_bot_v1_gru_1_while_identity&
"arbs_bot_v1_gru_1_while_identity_1&
"arbs_bot_v1_gru_1_while_identity_2&
"arbs_bot_v1_gru_1_while_identity_3&
"arbs_bot_v1_gru_1_while_identity_4=
9arbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_strided_slice_1y
uarbs_bot_v1_gru_1_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_1_tensorarrayunstack_tensorlistfromtensorM
:arbs_bot_v1_gru_1_while_gru_cell_1_readvariableop_resource:	U
Aarbs_bot_v1_gru_1_while_gru_cell_1_matmul_readvariableop_resource:
W
Carbs_bot_v1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource:
¢8arbs_bot_v1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp¢:arbs_bot_v1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp¢1arbs_bot_v1/gru_1/while/gru_cell_1/ReadVariableOp
Iarbs_bot_v1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
;arbs_bot_v1/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwarbs_bot_v1_gru_1_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_1_tensorarrayunstack_tensorlistfromtensor_0#arbs_bot_v1_gru_1_while_placeholderRarbs_bot_v1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¯
1arbs_bot_v1/gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp<arbs_bot_v1_gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0§
*arbs_bot_v1/gru_1/while/gru_cell_1/unstackUnpack9arbs_bot_v1/gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num¾
8arbs_bot_v1/gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOpCarbs_bot_v1_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ì
)arbs_bot_v1/gru_1/while/gru_cell_1/MatMulMatMulBarbs_bot_v1/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0@arbs_bot_v1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*arbs_bot_v1/gru_1/while/gru_cell_1/BiasAddBiasAdd3arbs_bot_v1/gru_1/while/gru_cell_1/MatMul:product:03arbs_bot_v1/gru_1/while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
2arbs_bot_v1/gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
(arbs_bot_v1/gru_1/while/gru_cell_1/splitSplit;arbs_bot_v1/gru_1/while/gru_cell_1/split/split_dim:output:03arbs_bot_v1/gru_1/while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitÂ
:arbs_bot_v1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpEarbs_bot_v1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ó
+arbs_bot_v1/gru_1/while/gru_cell_1/MatMul_1MatMul%arbs_bot_v1_gru_1_while_placeholder_2Barbs_bot_v1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
,arbs_bot_v1/gru_1/while/gru_cell_1/BiasAdd_1BiasAdd5arbs_bot_v1/gru_1/while/gru_cell_1/MatMul_1:product:03arbs_bot_v1/gru_1/while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
(arbs_bot_v1/gru_1/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ
4arbs_bot_v1/gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÕ
*arbs_bot_v1/gru_1/while/gru_cell_1/split_1SplitV5arbs_bot_v1/gru_1/while/gru_cell_1/BiasAdd_1:output:01arbs_bot_v1/gru_1/while/gru_cell_1/Const:output:0=arbs_bot_v1/gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitÊ
&arbs_bot_v1/gru_1/while/gru_cell_1/addAddV21arbs_bot_v1/gru_1/while/gru_cell_1/split:output:03arbs_bot_v1/gru_1/while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*arbs_bot_v1/gru_1/while/gru_cell_1/SigmoidSigmoid*arbs_bot_v1/gru_1/while/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
(arbs_bot_v1/gru_1/while/gru_cell_1/add_1AddV21arbs_bot_v1/gru_1/while/gru_cell_1/split:output:13arbs_bot_v1/gru_1/while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,arbs_bot_v1/gru_1/while/gru_cell_1/Sigmoid_1Sigmoid,arbs_bot_v1/gru_1/while/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
&arbs_bot_v1/gru_1/while/gru_cell_1/mulMul0arbs_bot_v1/gru_1/while/gru_cell_1/Sigmoid_1:y:03arbs_bot_v1/gru_1/while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
(arbs_bot_v1/gru_1/while/gru_cell_1/add_2AddV21arbs_bot_v1/gru_1/while/gru_cell_1/split:output:2*arbs_bot_v1/gru_1/while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'arbs_bot_v1/gru_1/while/gru_cell_1/ReluRelu,arbs_bot_v1/gru_1/while/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
(arbs_bot_v1/gru_1/while/gru_cell_1/mul_1Mul.arbs_bot_v1/gru_1/while/gru_cell_1/Sigmoid:y:0%arbs_bot_v1_gru_1_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(arbs_bot_v1/gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ã
&arbs_bot_v1/gru_1/while/gru_cell_1/subSub1arbs_bot_v1/gru_1/while/gru_cell_1/sub/x:output:0.arbs_bot_v1/gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
(arbs_bot_v1/gru_1/while/gru_cell_1/mul_2Mul*arbs_bot_v1/gru_1/while/gru_cell_1/sub:z:05arbs_bot_v1/gru_1/while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
(arbs_bot_v1/gru_1/while/gru_cell_1/add_3AddV2,arbs_bot_v1/gru_1/while/gru_cell_1/mul_1:z:0,arbs_bot_v1/gru_1/while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Barbs_bot_v1/gru_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ³
<arbs_bot_v1/gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%arbs_bot_v1_gru_1_while_placeholder_1Karbs_bot_v1/gru_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0,arbs_bot_v1/gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ_
arbs_bot_v1/gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
arbs_bot_v1/gru_1/while/addAddV2#arbs_bot_v1_gru_1_while_placeholder&arbs_bot_v1/gru_1/while/add/y:output:0*
T0*
_output_shapes
: a
arbs_bot_v1/gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¯
arbs_bot_v1/gru_1/while/add_1AddV2<arbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_loop_counter(arbs_bot_v1/gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
 arbs_bot_v1/gru_1/while/IdentityIdentity!arbs_bot_v1/gru_1/while/add_1:z:0^arbs_bot_v1/gru_1/while/NoOp*
T0*
_output_shapes
: ²
"arbs_bot_v1/gru_1/while/Identity_1IdentityBarbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_while_maximum_iterations^arbs_bot_v1/gru_1/while/NoOp*
T0*
_output_shapes
: 
"arbs_bot_v1/gru_1/while/Identity_2Identityarbs_bot_v1/gru_1/while/add:z:0^arbs_bot_v1/gru_1/while/NoOp*
T0*
_output_shapes
: ¼
"arbs_bot_v1/gru_1/while/Identity_3IdentityLarbs_bot_v1/gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^arbs_bot_v1/gru_1/while/NoOp*
T0*
_output_shapes
: ®
"arbs_bot_v1/gru_1/while/Identity_4Identity,arbs_bot_v1/gru_1/while/gru_cell_1/add_3:z:0^arbs_bot_v1/gru_1/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
arbs_bot_v1/gru_1/while/NoOpNoOp9^arbs_bot_v1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp;^arbs_bot_v1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp2^arbs_bot_v1/gru_1/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "x
9arbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_strided_slice_1;arbs_bot_v1_gru_1_while_arbs_bot_v1_gru_1_strided_slice_1_0"
Carbs_bot_v1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resourceEarbs_bot_v1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"
Aarbs_bot_v1_gru_1_while_gru_cell_1_matmul_readvariableop_resourceCarbs_bot_v1_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"z
:arbs_bot_v1_gru_1_while_gru_cell_1_readvariableop_resource<arbs_bot_v1_gru_1_while_gru_cell_1_readvariableop_resource_0"M
 arbs_bot_v1_gru_1_while_identity)arbs_bot_v1/gru_1/while/Identity:output:0"Q
"arbs_bot_v1_gru_1_while_identity_1+arbs_bot_v1/gru_1/while/Identity_1:output:0"Q
"arbs_bot_v1_gru_1_while_identity_2+arbs_bot_v1/gru_1/while/Identity_2:output:0"Q
"arbs_bot_v1_gru_1_while_identity_3+arbs_bot_v1/gru_1/while/Identity_3:output:0"Q
"arbs_bot_v1_gru_1_while_identity_4+arbs_bot_v1/gru_1/while/Identity_4:output:0"ð
uarbs_bot_v1_gru_1_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_1_tensorarrayunstack_tensorlistfromtensorwarbs_bot_v1_gru_1_while_tensorarrayv2read_tensorlistgetitem_arbs_bot_v1_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2t
8arbs_bot_v1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp8arbs_bot_v1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp2x
:arbs_bot_v1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:arbs_bot_v1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp2f
1arbs_bot_v1/gru_1/while/gru_cell_1/ReadVariableOp1arbs_bot_v1/gru_1/while/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
þ
¶
$__inference_gru_layer_call_fn_508387

inputs
unknown:	
	unknown_0:	
	unknown_1:

identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_506799t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú¹
ô	
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_508030

inputs7
$gru_gru_cell_readvariableop_resource:	>
+gru_gru_cell_matmul_readvariableop_resource:	A
-gru_gru_cell_matmul_1_readvariableop_resource:
;
(gru_1_gru_cell_1_readvariableop_resource:	C
/gru_1_gru_cell_1_matmul_readvariableop_resource:
E
1gru_1_gru_cell_1_matmul_1_readvariableop_resource:
7
$dense_matmul_readvariableop_resource:	 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢"gru/gru_cell/MatMul/ReadVariableOp¢$gru/gru_cell/MatMul_1/ReadVariableOp¢gru/gru_cell/ReadVariableOp¢	gru/while¢&gru_1/gru_cell_1/MatMul/ReadVariableOp¢(gru_1/gru_cell_1/MatMul_1/ReadVariableOp¢gru_1/gru_cell_1/ReadVariableOp¢gru_1/while?
	gru/ShapeShapeinputs*
T0*
_output_shapes
:a
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    y
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:c
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒc
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ý
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0{
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÍ
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru/gru_cell/MatMul_1MatMulgru/zeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿi
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒJ
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿX
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *!
bodyR
gru_while_body_507770*!
condR
gru_while_cond_507769*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿe
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    N
gru_1/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
:c
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_1/transpose	Transposegru/transpose_1:y:0gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
:e
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_1/gru_cell_1/ReadVariableOpReadVariableOp(gru_1_gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_1/gru_cell_1/unstackUnpack'gru_1/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¤
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/BiasAddBiasAdd!gru_1/gru_cell_1/MatMul:product:0!gru_1/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_1/gru_cell_1/splitSplit)gru_1/gru_cell_1/split/split_dim:output:0!gru_1/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
(gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_1/gru_cell_1/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
gru_1/gru_cell_1/BiasAdd_1BiasAdd#gru_1/gru_cell_1/MatMul_1:product:0!gru_1/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
gru_1/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿm
"gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/split_1SplitV#gru_1/gru_cell_1/BiasAdd_1:output:0gru_1/gru_cell_1/Const:output:0+gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_1/gru_cell_1/addAddV2gru_1/gru_cell_1/split:output:0!gru_1/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/add_1AddV2gru_1/gru_cell_1/split:output:1!gru_1/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/mulMulgru_1/gru_cell_1/Sigmoid_1:y:0!gru_1/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/add_2AddV2gru_1/gru_cell_1/split:output:2gru_1/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
gru_1/gru_cell_1/ReluRelugru_1/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/mul_1Mulgru_1/gru_cell_1/Sigmoid:y:0gru_1/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/mul_2Mulgru_1/gru_cell_1/sub:z:0#gru_1/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_1:z:0gru_1/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   d
"gru_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :×
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0+gru_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_1_readvariableop_resource/gru_1_gru_cell_1_matmul_readvariableop_resource1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_1_while_body_507920*#
condR
gru_1_while_cond_507919*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   é
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsn
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
dense/MatMulMatMulgru_1/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while'^gru_1/gru_cell_1/MatMul/ReadVariableOp)^gru_1/gru_cell_1/MatMul_1/ReadVariableOp ^gru_1/gru_cell_1/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while2P
&gru_1/gru_cell_1/MatMul/ReadVariableOp&gru_1/gru_cell_1/MatMul/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_1/ReadVariableOp(gru_1/gru_cell_1/MatMul_1/ReadVariableOp2B
gru_1/gru_cell_1/ReadVariableOpgru_1/gru_cell_1/ReadVariableOp2
gru_1/whilegru_1/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

»
&__inference_gru_1_layer_call_fn_509021
inputs_0
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_506447p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
Ü
ª
while_cond_506709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_506709___redundant_placeholder04
0while_while_cond_506709___redundant_placeholder14
0while_while_cond_506709___redundant_placeholder24
0while_while_cond_506709___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
í=

while_body_509429
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_1_readvariableop_resource_0:	E
1while_gru_cell_1_matmul_readvariableop_resource_0:
G
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_1_readvariableop_resource:	C
/while_gru_cell_1_matmul_readvariableop_resource:
E
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ë
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ü
ª
while_cond_509118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_509118___redundant_placeholder04
0while_while_cond_509118___redundant_placeholder14
0while_while_cond_509118___redundant_placeholder24
0while_while_cond_509118___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
íM

A__inference_gru_1_layer_call_and_return_conditional_losses_509674

inputs5
"gru_cell_1_readvariableop_resource:	=
)gru_cell_1_matmul_readvariableop_resource:
?
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_509584*
condR
while_cond_509583*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ	
ô
C__inference_dense_2_layer_call_and_return_conditional_losses_509733

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ	
ô
C__inference_dense_2_layer_call_and_return_conditional_losses_507013

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507582
input_1

gru_507552:	

gru_507554:	

gru_507556:

gru_1_507559:	 
gru_1_507561:
 
gru_1_507563:

dense_507566:	 
dense_507568:  
dense_1_507571: 
dense_1_507573: 
dense_2_507576:
dense_2_507578:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCallð
gru/StatefulPartitionedCallStatefulPartitionedCallinput_1
gru_507552
gru_507554
gru_507556*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_506799
gru_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0gru_1_507559gru_1_507561gru_1_507563*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_1_layer_call_and_return_conditional_losses_506961
dense/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0dense_507566dense_507568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506980
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_507571dense_1_507573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_506997
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_507576dense_2_507578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_507013w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ÅK
ô
?__inference_gru_layer_call_and_return_conditional_losses_509010

inputs3
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	=
)gru_cell_matmul_1_readvariableop_resource:

identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÁ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split|
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_508921*
condR
while_cond_508920*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
ª
while_cond_508767
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_508767___redundant_placeholder04
0while_while_cond_508767___redundant_placeholder14
0while_while_cond_508767___redundant_placeholder24
0while_while_cond_508767___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÅK
ô
?__inference_gru_layer_call_and_return_conditional_losses_506799

inputs3
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	=
)gru_cell_matmul_1_readvariableop_resource:

identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÁ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split|
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_506710*
condR
while_cond_506709*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ

gru_1_while_cond_507919(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1@
<gru_1_while_gru_1_while_cond_507919___redundant_placeholder0@
<gru_1_while_gru_1_while_cond_507919___redundant_placeholder1@
<gru_1_while_gru_1_while_cond_507919___redundant_placeholder2@
<gru_1_while_gru_1_while_cond_507919___redundant_placeholder3
gru_1_while_identity
z
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
È
ö
gru_while_cond_507769$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1<
8gru_while_gru_while_cond_507769___redundant_placeholder0<
8gru_while_gru_while_cond_507769___redundant_placeholder1<
8gru_while_gru_while_cond_507769___redundant_placeholder2<
8gru_while_gru_while_cond_507769___redundant_placeholder3
gru_while_identity
r
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
í=

while_body_509584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_1_readvariableop_resource_0:	E
1while_gru_cell_1_matmul_readvariableop_resource_0:
G
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_1_readvariableop_resource:	C
/while_gru_cell_1_matmul_readvariableop_resource:
E
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¶
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ë
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
£;
ï
while_body_508615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	E
1while_gru_cell_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	C
/while_gru_cell_matmul_1_readvariableop_resource:
¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ì3
ö
?__inference_gru_layer_call_and_return_conditional_losses_506107

inputs"
gru_cell_506031:	"
gru_cell_506033:	#
gru_cell_506035:

identity¢ gru_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¼
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_506031gru_cell_506033gru_cell_506035*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_506030n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_506031gru_cell_506033gru_cell_506035*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_506043*
condR
while_cond_506042*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¿
,__inference_arbs_bot_v1_layer_call_fn_507549
input_1
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ü
ª
while_cond_509273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_509273___redundant_placeholder04
0while_while_cond_509273___redundant_placeholder14
0while_while_cond_509273___redundant_placeholder24
0while_while_cond_509273___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
²

"__inference__traced_restore_510230
file_prefix0
assignvariableop_dense_kernel:	 +
assignvariableop_1_dense_bias: 3
!assignvariableop_2_dense_1_kernel: -
assignvariableop_3_dense_1_bias:3
!assignvariableop_4_dense_2_kernel:-
assignvariableop_5_dense_2_bias:9
&assignvariableop_6_gru_gru_cell_kernel:	D
0assignvariableop_7_gru_gru_cell_recurrent_kernel:
7
$assignvariableop_8_gru_gru_cell_bias:	>
*assignvariableop_9_gru_1_gru_cell_1_kernel:
I
5assignvariableop_10_gru_1_gru_cell_1_recurrent_kernel:
<
)assignvariableop_11_gru_1_gru_cell_1_bias:	'
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: A
.assignvariableop_14_adam_m_gru_gru_cell_kernel:	A
.assignvariableop_15_adam_v_gru_gru_cell_kernel:	L
8assignvariableop_16_adam_m_gru_gru_cell_recurrent_kernel:
L
8assignvariableop_17_adam_v_gru_gru_cell_recurrent_kernel:
?
,assignvariableop_18_adam_m_gru_gru_cell_bias:	?
,assignvariableop_19_adam_v_gru_gru_cell_bias:	F
2assignvariableop_20_adam_m_gru_1_gru_cell_1_kernel:
F
2assignvariableop_21_adam_v_gru_1_gru_cell_1_kernel:
P
<assignvariableop_22_adam_m_gru_1_gru_cell_1_recurrent_kernel:
P
<assignvariableop_23_adam_v_gru_1_gru_cell_1_recurrent_kernel:
C
0assignvariableop_24_adam_m_gru_1_gru_cell_1_bias:	C
0assignvariableop_25_adam_v_gru_1_gru_cell_1_bias:	:
'assignvariableop_26_adam_m_dense_kernel:	 :
'assignvariableop_27_adam_v_dense_kernel:	 3
%assignvariableop_28_adam_m_dense_bias: 3
%assignvariableop_29_adam_v_dense_bias: ;
)assignvariableop_30_adam_m_dense_1_kernel: ;
)assignvariableop_31_adam_v_dense_1_kernel: 5
'assignvariableop_32_adam_m_dense_1_bias:5
'assignvariableop_33_adam_v_dense_1_bias:;
)assignvariableop_34_adam_m_dense_2_kernel:;
)assignvariableop_35_adam_v_dense_2_kernel:5
'assignvariableop_36_adam_m_dense_2_bias:5
'assignvariableop_37_adam_v_dense_2_bias:%
assignvariableop_38_total_1: %
assignvariableop_39_count_1: #
assignvariableop_40_total: #
assignvariableop_41_count: 
identity_43¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ý
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*
valueùBö+B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÆ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ø
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_6AssignVariableOp&assignvariableop_6_gru_gru_cell_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_7AssignVariableOp0assignvariableop_7_gru_gru_cell_recurrent_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_8AssignVariableOp$assignvariableop_8_gru_gru_cell_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_9AssignVariableOp*assignvariableop_9_gru_1_gru_cell_1_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_10AssignVariableOp5assignvariableop_10_gru_1_gru_cell_1_recurrent_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_11AssignVariableOp)assignvariableop_11_gru_1_gru_cell_1_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:¶
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterationIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_14AssignVariableOp.assignvariableop_14_adam_m_gru_gru_cell_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_15AssignVariableOp.assignvariableop_15_adam_v_gru_gru_cell_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ñ
AssignVariableOp_16AssignVariableOp8assignvariableop_16_adam_m_gru_gru_cell_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ñ
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_v_gru_gru_cell_recurrent_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_m_gru_gru_cell_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_v_gru_gru_cell_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_m_gru_1_gru_cell_1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_v_gru_1_gru_cell_1_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_22AssignVariableOp<assignvariableop_22_adam_m_gru_1_gru_cell_1_recurrent_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_23AssignVariableOp<assignvariableop_23_adam_v_gru_1_gru_cell_1_recurrent_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_m_gru_1_gru_cell_1_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_v_gru_1_gru_cell_1_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_m_dense_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_v_dense_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_m_dense_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_v_dense_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_m_dense_1_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_v_dense_1_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_m_dense_1_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_v_dense_1_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_m_dense_2_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_v_dense_2_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_m_dense_2_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_v_dense_2_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_1Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ë
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: Ø
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ü
ª
while_cond_506565
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_506565___redundant_placeholder04
0while_while_cond_506565___redundant_placeholder14
0while_while_cond_506565___redundant_placeholder24
0while_while_cond_506565___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¤
Ú
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_506368

inputs

states*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
?
input_14
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ;
dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
©
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell
 
state_spec"
_tf_keras_rnn_layer
»
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
»
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
»
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
v
90
:1
;2
<3
=4
>5
'6
(7
/8
09
710
811"
trackable_list_wrapper
v
90
:1
;2
<3
=4
>5
'6
(7
/8
09
710
811"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
å
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_32ú
,__inference_arbs_bot_v1_layer_call_fn_507047
,__inference_arbs_bot_v1_layer_call_fn_507677
,__inference_arbs_bot_v1_layer_call_fn_507706
,__inference_arbs_bot_v1_layer_call_fn_507549¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zDtrace_0zEtrace_1zFtrace_2zGtrace_3
Ñ
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32æ
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_508030
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_508354
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507582
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507615¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
ÌBÉ
!__inference__wrapped_model_505960input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

L
_variables
M_iterations
N_learning_rate
O_index_dict
P
_momentums
Q_velocities
R_update_step_xla"
experimentalOptimizer
,
Sserving_default"
signature_map
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ú
Ztrace_0
[trace_1
\trace_2
]trace_32ï
$__inference_gru_layer_call_fn_508365
$__inference_gru_layer_call_fn_508376
$__inference_gru_layer_call_fn_508387
$__inference_gru_layer_call_fn_508398Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zZtrace_0z[trace_1z\trace_2z]trace_3
Æ
^trace_0
_trace_1
`trace_2
atrace_32Û
?__inference_gru_layer_call_and_return_conditional_losses_508551
?__inference_gru_layer_call_and_return_conditional_losses_508704
?__inference_gru_layer_call_and_return_conditional_losses_508857
?__inference_gru_layer_call_and_return_conditional_losses_509010Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z^trace_0z_trace_1z`trace_2zatrace_3
"
_generic_user_object
è
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator

9kernel
:recurrent_kernel
;bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

istates
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
â
otrace_0
ptrace_1
qtrace_2
rtrace_32÷
&__inference_gru_1_layer_call_fn_509021
&__inference_gru_1_layer_call_fn_509032
&__inference_gru_1_layer_call_fn_509043
&__inference_gru_1_layer_call_fn_509054Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0zptrace_1zqtrace_2zrtrace_3
Î
strace_0
ttrace_1
utrace_2
vtrace_32ã
A__inference_gru_1_layer_call_and_return_conditional_losses_509209
A__inference_gru_1_layer_call_and_return_conditional_losses_509364
A__inference_gru_1_layer_call_and_return_conditional_losses_509519
A__inference_gru_1_layer_call_and_return_conditional_losses_509674Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zstrace_0zttrace_1zutrace_2zvtrace_3
"
_generic_user_object
è
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
}_random_generator

<kernel
=recurrent_kernel
>bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
ì
trace_02Í
&__inference_dense_layer_call_fn_509683¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02è
A__inference_dense_layer_call_and_return_conditional_losses_509694¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
:	 2dense/kernel
: 2
dense/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_dense_1_layer_call_fn_509703¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_dense_1_layer_call_and_return_conditional_losses_509714¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 : 2dense_1/kernel
:2dense_1/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_dense_2_layer_call_fn_509723¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_dense_2_layer_call_and_return_conditional_losses_509733¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 :2dense_2/kernel
:2dense_2/bias
&:$	2gru/gru_cell/kernel
1:/
2gru/gru_cell/recurrent_kernel
$:"	2gru/gru_cell/bias
+:)
2gru_1/gru_cell_1/kernel
5:3
2!gru_1/gru_cell_1/recurrent_kernel
(:&	2gru_1/gru_cell_1/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
,__inference_arbs_bot_v1_layer_call_fn_507047input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
,__inference_arbs_bot_v1_layer_call_fn_507677inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
,__inference_arbs_bot_v1_layer_call_fn_507706inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
,__inference_arbs_bot_v1_layer_call_fn_507549input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_508030inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_508354inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507582input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507615input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö
M0
1
2
3
4
5
6
7
8
9
10
11
 12
¡13
¢14
£15
¤16
¥17
¦18
§19
¨20
©21
ª22
«23
¬24"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper

0
1
2
3
4
5
¡6
£7
¥8
§9
©10
«11"
trackable_list_wrapper

0
1
2
3
4
 5
¢6
¤7
¦8
¨9
ª10
¬11"
trackable_list_wrapper
¿2¼¹
®²ª
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
ËBÈ
$__inference_signature_wrapper_507648input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
$__inference_gru_layer_call_fn_508365inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
$__inference_gru_layer_call_fn_508376inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
$__inference_gru_layer_call_fn_508387inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
$__inference_gru_layer_call_fn_508398inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
§B¤
?__inference_gru_layer_call_and_return_conditional_losses_508551inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
§B¤
?__inference_gru_layer_call_and_return_conditional_losses_508704inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¥B¢
?__inference_gru_layer_call_and_return_conditional_losses_508857inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¥B¢
?__inference_gru_layer_call_and_return_conditional_losses_509010inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
Ñ
²trace_0
³trace_12
)__inference_gru_cell_layer_call_fn_509747
)__inference_gru_cell_layer_call_fn_509761½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z²trace_0z³trace_1

´trace_0
µtrace_12Ì
D__inference_gru_cell_layer_call_and_return_conditional_losses_509800
D__inference_gru_cell_layer_call_and_return_conditional_losses_509839½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z´trace_0zµtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
&__inference_gru_1_layer_call_fn_509021inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
&__inference_gru_1_layer_call_fn_509032inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
&__inference_gru_1_layer_call_fn_509043inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
&__inference_gru_1_layer_call_fn_509054inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
©B¦
A__inference_gru_1_layer_call_and_return_conditional_losses_509209inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
©B¦
A__inference_gru_1_layer_call_and_return_conditional_losses_509364inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
§B¤
A__inference_gru_1_layer_call_and_return_conditional_losses_509519inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
§B¤
A__inference_gru_1_layer_call_and_return_conditional_losses_509674inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
Õ
»trace_0
¼trace_12
+__inference_gru_cell_1_layer_call_fn_509853
+__inference_gru_cell_1_layer_call_fn_509867½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z»trace_0z¼trace_1

½trace_0
¾trace_12Ð
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_509906
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_509945½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z½trace_0z¾trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
&__inference_dense_layer_call_fn_509683inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õBò
A__inference_dense_layer_call_and_return_conditional_losses_509694inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_1_layer_call_fn_509703inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_dense_1_layer_call_and_return_conditional_losses_509714inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_2_layer_call_fn_509723inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_dense_2_layer_call_and_return_conditional_losses_509733inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
¿	variables
À	keras_api

Átotal

Âcount"
_tf_keras_metric
c
Ã	variables
Ä	keras_api

Åtotal

Æcount
Ç
_fn_kwargs"
_tf_keras_metric
+:)	2Adam/m/gru/gru_cell/kernel
+:)	2Adam/v/gru/gru_cell/kernel
6:4
2$Adam/m/gru/gru_cell/recurrent_kernel
6:4
2$Adam/v/gru/gru_cell/recurrent_kernel
):'	2Adam/m/gru/gru_cell/bias
):'	2Adam/v/gru/gru_cell/bias
0:.
2Adam/m/gru_1/gru_cell_1/kernel
0:.
2Adam/v/gru_1/gru_cell_1/kernel
::8
2(Adam/m/gru_1/gru_cell_1/recurrent_kernel
::8
2(Adam/v/gru_1/gru_cell_1/recurrent_kernel
-:+	2Adam/m/gru_1/gru_cell_1/bias
-:+	2Adam/v/gru_1/gru_cell_1/bias
$:"	 2Adam/m/dense/kernel
$:"	 2Adam/v/dense/kernel
: 2Adam/m/dense/bias
: 2Adam/v/dense/bias
%:# 2Adam/m/dense_1/kernel
%:# 2Adam/v/dense_1/kernel
:2Adam/m/dense_1/bias
:2Adam/v/dense_1/bias
%:#2Adam/m/dense_2/kernel
%:#2Adam/v/dense_2/kernel
:2Adam/m/dense_2/bias
:2Adam/v/dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bÿ
)__inference_gru_cell_layer_call_fn_509747inputsstates_0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
)__inference_gru_cell_layer_call_fn_509761inputsstates_0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_gru_cell_layer_call_and_return_conditional_losses_509800inputsstates_0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_gru_cell_layer_call_and_return_conditional_losses_509839inputsstates_0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
+__inference_gru_cell_1_layer_call_fn_509853inputsstates_0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
+__inference_gru_cell_1_layer_call_fn_509867inputsstates_0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_509906inputsstates_0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_509945inputsstates_0"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
Á0
Â1"
trackable_list_wrapper
.
¿	variables"
_generic_user_object
:  (2total
:  (2count
0
Å0
Æ1"
trackable_list_wrapper
.
Ã	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
!__inference__wrapped_model_505960w;9:><='(/0784¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿÅ
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507582z;9:><='(/078<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_507615z;9:><='(/078<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Ä
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_508030y;9:><='(/078;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Ä
G__inference_arbs_bot_v1_layer_call_and_return_conditional_losses_508354y;9:><='(/078;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
,__inference_arbs_bot_v1_layer_call_fn_507047o;9:><='(/078<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
,__inference_arbs_bot_v1_layer_call_fn_507549o;9:><='(/078<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
,__inference_arbs_bot_v1_layer_call_fn_507677n;9:><='(/078;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
,__inference_arbs_bot_v1_layer_call_fn_507706n;9:><='(/078;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿª
C__inference_dense_1_layer_call_and_return_conditional_losses_509714c/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
(__inference_dense_1_layer_call_fn_509703X/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "!
unknownÿÿÿÿÿÿÿÿÿª
C__inference_dense_2_layer_call_and_return_conditional_losses_509733c78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
(__inference_dense_2_layer_call_fn_509723X78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿ©
A__inference_dense_layer_call_and_return_conditional_losses_509694d'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ 
 
&__inference_dense_layer_call_fn_509683Y'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿ Ì
A__inference_gru_1_layer_call_and_return_conditional_losses_509209><=P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 Ì
A__inference_gru_1_layer_call_and_return_conditional_losses_509364><=P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 »
A__inference_gru_1_layer_call_and_return_conditional_losses_509519v><=@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 »
A__inference_gru_1_layer_call_and_return_conditional_losses_509674v><=@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¥
&__inference_gru_1_layer_call_fn_509021{><=P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ""
unknownÿÿÿÿÿÿÿÿÿ¥
&__inference_gru_1_layer_call_fn_509032{><=P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª ""
unknownÿÿÿÿÿÿÿÿÿ
&__inference_gru_1_layer_call_fn_509043k><=@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ""
unknownÿÿÿÿÿÿÿÿÿ
&__inference_gru_1_layer_call_fn_509054k><=@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª ""
unknownÿÿÿÿÿÿÿÿÿ
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_509906É><=^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states_0ÿÿÿÿÿÿÿÿÿ
p 
ª "b¢_
X¢U
%"

tensor_0_0ÿÿÿÿÿÿÿÿÿ
,)
'$
tensor_0_1_0ÿÿÿÿÿÿÿÿÿ
 
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_509945É><=^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states_0ÿÿÿÿÿÿÿÿÿ
p
ª "b¢_
X¢U
%"

tensor_0_0ÿÿÿÿÿÿÿÿÿ
,)
'$
tensor_0_1_0ÿÿÿÿÿÿÿÿÿ
 ë
+__inference_gru_cell_1_layer_call_fn_509853»><=^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states_0ÿÿÿÿÿÿÿÿÿ
p 
ª "T¢Q
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
*'
%"

tensor_1_0ÿÿÿÿÿÿÿÿÿë
+__inference_gru_cell_1_layer_call_fn_509867»><=^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states_0ÿÿÿÿÿÿÿÿÿ
p
ª "T¢Q
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
*'
%"

tensor_1_0ÿÿÿÿÿÿÿÿÿ
D__inference_gru_cell_layer_call_and_return_conditional_losses_509800È;9:]¢Z
S¢P
 
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states_0ÿÿÿÿÿÿÿÿÿ
p 
ª "b¢_
X¢U
%"

tensor_0_0ÿÿÿÿÿÿÿÿÿ
,)
'$
tensor_0_1_0ÿÿÿÿÿÿÿÿÿ
 
D__inference_gru_cell_layer_call_and_return_conditional_losses_509839È;9:]¢Z
S¢P
 
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states_0ÿÿÿÿÿÿÿÿÿ
p
ª "b¢_
X¢U
%"

tensor_0_0ÿÿÿÿÿÿÿÿÿ
,)
'$
tensor_0_1_0ÿÿÿÿÿÿÿÿÿ
 è
)__inference_gru_cell_layer_call_fn_509747º;9:]¢Z
S¢P
 
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states_0ÿÿÿÿÿÿÿÿÿ
p 
ª "T¢Q
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
*'
%"

tensor_1_0ÿÿÿÿÿÿÿÿÿè
)__inference_gru_cell_layer_call_fn_509761º;9:]¢Z
S¢P
 
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states_0ÿÿÿÿÿÿÿÿÿ
p
ª "T¢Q
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
*'
%"

tensor_1_0ÿÿÿÿÿÿÿÿÿÖ
?__inference_gru_layer_call_and_return_conditional_losses_508551;9:O¢L
E¢B
41
/,
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ":¢7
0-
tensor_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ö
?__inference_gru_layer_call_and_return_conditional_losses_508704;9:O¢L
E¢B
41
/,
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª ":¢7
0-
tensor_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¼
?__inference_gru_layer_call_and_return_conditional_losses_508857y;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "1¢.
'$
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¼
?__inference_gru_layer_call_and_return_conditional_losses_509010y;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "1¢.
'$
tensor_0ÿÿÿÿÿÿÿÿÿ
 °
$__inference_gru_layer_call_fn_508365;9:O¢L
E¢B
41
/,
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "/,
unknownÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
$__inference_gru_layer_call_fn_508376;9:O¢L
E¢B
41
/,
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "/,
unknownÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$__inference_gru_layer_call_fn_508387n;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#
unknownÿÿÿÿÿÿÿÿÿ
$__inference_gru_layer_call_fn_508398n;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#
unknownÿÿÿÿÿÿÿÿÿ«
$__inference_signature_wrapper_507648;9:><='(/078?¢<
¢ 
5ª2
0
input_1%"
input_1ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ