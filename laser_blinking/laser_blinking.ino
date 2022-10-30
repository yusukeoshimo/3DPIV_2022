#define LASER1 7
#define LASER2 8

int turn_on_time = 1000;
int split_time = 3;
int cooldown_time = 100;

void setup() {
  pinMode(LASER1, OUTPUT);
  pinMode(LASER2, OUTPUT);
  digitalWrite(LASER1,LOW);
  digitalWrite(LASER2,LOW);
}

void loop() {
  delay(cooldown_time); 
  
  digitalWrite(LASER1,HIGH);
  delay(turn_on_time);
  digitalWrite(LASER1,LOW);

  //delay(split_time);

  digitalWrite(LASER2,HIGH);
  delay(turn_on_time);
  digitalWrite(LASER2,LOW);
}
