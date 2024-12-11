extends Control

'''quiero implementar esto para ir mas rapido o consumir menos recursos'''

func _ready() -> void:

	var byte_array : PackedByteArray
	var float_value = 1.2345
	var byte_offset = 2



	byte_array = PackedByteArray()
	byte_array.resize(10)  

	byte_array.encode_half(byte_offset, float_value)
	var decoded_value = byte_array.decode_half(byte_offset)
	
	print("Original value:", float_value)
	print("Decoded value:", decoded_value)
	prints("Original array :)",byte_array)

	# prueba :)
	for i in range (20):
		signo8(randf_range(0.9, 0.0001) + randf_range(0.0001, 0.0))





# 8 bit signio 1 - 1
func encode_minifloat(value: float) -> int:
	if value == 0.0:
		return 0
	var sign = 0
	if value < 0:
		sign = 1
		value = -value
	var mantissa = int(value * 127) & 0x7F
	return (sign << 7) | mantissa


# 8 bit signo 1 -1
func decode_minifloat(encoded_value: int) -> float:
	if encoded_value == 0:
		return 0.0
	var sign = (encoded_value >> 7) & 0x01
	var mantissa = (encoded_value & 0x7F) / 127.0
	if sign == 1:
		return -mantissa
	return mantissa



# 8 bits sin signo
func encode_minifloat_unsigned(value: float) -> int:

	if value == 0.0:
		return 0
	var mantissa = int(value * 255) & 0xFF
	return mantissa


# decodificar 8 bits sin signo
func decode_minifloat_unsigned(encoded_value: int) -> float:

	if encoded_value == 0:
		return 0.0
	var mantissa = encoded_value & 0xFF
	return mantissa / 255.0








# es como bit sin signo a tope max 1 
func encode_minifloat_unsigned_limited(value: float) -> int:
	value = clamp(value, 0.0, 1.0)
	var exponent = 0  # Usar exponente fijo de 0
	var mantissa = int(value * 255 + 0.5) & 0xFF  
	return (exponent << 7) | mantissa

# es como bit sin signo a tope max 1 
func decode_minifloat_unsigned_limited(encoded_value: int) -> float:
	var exponent = 0 
	var mantissa = (encoded_value & 0xFF) / 255.0  
	return mantissa * pow(2, exponent)














func signo8(fp8 : float):


	
	var float_value = fp8
	var encoded_value = encode_minifloat(float_value)
	var decoded_value = decode_minifloat(encoded_value)

	print("Original value:", float_value)
	print("Encoded value 8s:", encoded_value)
	print("Decoded value 8s:", decoded_value)
	


	encoded_value = encode_minifloat_unsigned(float_value)
	decoded_value = decode_minifloat_unsigned(encoded_value)

	print("Original value:", float_value)
	print("Encoded value 8u:", encoded_value)
	print("Decoded value 8u:", decoded_value)
	
	
	'''
	maximo: 1
	minimo distinto a 0 : 0.00392157
	minimo : 0
	muy paresido a encode_minifloat_unsigned creo que es mas rapido 
	'''

	var encoded_value_8u_limited = encode_minifloat_unsigned_limited(float_value)
	var decoded_value_8u_limited = decode_minifloat_unsigned_limited(encoded_value_8u_limited)

	print("Original value:", float_value)
	print("Encoded value 8u limited:", encoded_value_8u_limited)
	print("Decoded value 8u limited:", decoded_value_8u_limited)

	
	
	
