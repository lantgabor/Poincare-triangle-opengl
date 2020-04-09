//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : ******
// Neptun : ******
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders

bool intersect(vec2 p1, vec2 p2, vec2 r0, vec2 r1) {
	vec2 v = vec2(r1.x - r0.x, r1.y - r0.y);
	vec2 n = vec2(v.y, -1.0f * v.x);

	float t1 = dot(n, (p1 - r0)) * dot(n, (p2 - r0));
	
	v = vec2(p2.x - p1.x, p2.y - p1.y);
	n = vec2(v.y, -1 * v.x);

	float t2 = dot(n, (r0 - p1)) * dot(n, (r1 - p1));
	
	return (t1 < 0 && t2 < 0);
}

vec2 inversePoint(vec2 a) {
	// Matematikai helyess�get igazol� magyar�zat, hogy mi�rt j� a feladat megold�s�hoz
	// az inverzi� az ortogon�lis k�r�k meghat�roz�s�n�l.
	// https://mathworld.wolfram.com/OrthogonalCircles.html

	float alpha = 1 / (powf(a.x, 2) + powf(a.y, 2));
	float x = alpha * a.x;
	float y = alpha * a.y;
	return vec2(x, y);
}

vec3 orthogonal(vec2 a, vec2 b, vec2 c) {
	// H�rom pontra vett k�r k�z�ppontj�nak meghat�roz�sa
	// http://mathforum.org/library/drmath/view/54323.html

	float temp = b.x * b.x + b.y * b.y;
	float bc = (a.x * a.x + a.y * a.y - temp) / 2;
	float cd = (temp - c.x * c.x - c.y * c.y) / 2;
	float det = (a.x - b.x) * (b.y - c.y) - (b.x - c.x) * (a.y - b.y);

	float cx = (bc * (b.y - c.y) - cd * (a.y - b.y)) / det;
	float cy = ((a.x - b.x) * cd - (b.x - c.x) * bc) / det;

	float radius = sqrtf(powf((cx - a.x), 2) + powf((cy - a.y), 2));
	return vec3(cx, cy, radius);
}

float siriusAngle(vec2 click, vec3 cPos1, vec3 cPos2, vec2 before, vec2 at, vec2 after) {
	vec2 bef = vec2(before.x - at.x, before.y - at.y);
	vec2 aft = vec2(after.x - at.x, after.y - at.y);

	float fi = acosf(dot(bef, aft) / (length(bef) * length(aft)));

	vec2 a = vec2(cPos1.x - click.x, cPos1.y - click.y);
	vec2 b = vec2(cPos2.x - click.x, cPos2.y - click.y);
	float alpha = acosf(dot(a, b) / (length(a) * length(b))) / M_PI * 180;
	return fi < M_PI ? fabs(alpha - 180) : fabs(alpha);
}

float siriusLength(vec2 p1, vec2 p2) {
	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;
	return sqrtf(dx * dx + dy * dy) / (1 - p2.x * p2.x - p2.y * p2.y);
}

struct Circle
{
	unsigned int vao, vbo;
	vec3 color;
	vec2 pos;
	float size;
	Circle(vec2 pos, vec3 color, float size = 1)
	{

		this->pos = pos;
		this->color = color;
		this->size = size;
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		vec2 vertices[100];
		for (size_t i = 0; i < 100; i++)
		{
			float fi = i * 2 * M_PI / 100;
			vertices[i] = vec2(cosf(fi), sinf(fi));
		}
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * 100,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	void Draw() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z); // 3 floats

		float MVPtransf[4][4] = { 1 * size, 0, 0, 0,    // MVP matrix, 
								  0, 1 * size, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  pos.x, pos.y, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 100 /*# Elements*/);
	}
};

std::vector<vec3> cPos;

struct TriangleSide {
	unsigned int vao, vbo;
	vec3 color;
	vec2 vertices[60];
	TriangleSide(vec2 p1, vec2 p2, vec2 p3, vec3 color)
	{
		this->color = color;
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		vec3 pos;

		for (size_t i = 0; i < 20; i++)
		{
			pos = orthogonal(inversePoint(p1), p1, p2);

			if(i==0) cPos.push_back(pos);

			float from = atan2(p1.y - pos.y, p1.x - pos.x);
			float to = atan2(p2.y - pos.y, p2.x - pos.x);

			float diff = to - from;

			if (diff < -M_PI) {
				diff += 2 * M_PI;
			}
			else if (diff > M_PI) {
				diff -= 2 * M_PI;
			}

			float fi = from + (diff * i / 20);

			vertices[i] = vec2(pos.x + pos.z * cosf(fi), pos.y + pos.z * sinf(fi));
		}

		for (size_t i = 20; i < 40; i++)
		{
			pos = orthogonal(inversePoint(p2), p2, p3);

			if(i == 20) cPos.push_back(pos);

			float from = atan2(p2.y - pos.y, p2.x - pos.x);
			float to = atan2(p3.y - pos.y, p3.x - pos.x);

			float diff = to - from;

			if (diff < -M_PI) {
				diff += 2 * M_PI;
			}
			else if (diff > M_PI) {
				diff -= 2 * M_PI;
			}

			float fi = from + (diff * (i - 20) / 20);

			vertices[i] = vec2(pos.x + pos.z * cosf(fi), pos.y + pos.z * sinf(fi));
		}

		for (size_t i = 40; i < 60; i++)
		{
			pos = orthogonal(inversePoint(p3), p3, p1);

			if(i==40) cPos.push_back(pos);

			float from = atan2(p3.y - pos.y, p3.x - pos.x);
			float to = atan2(p1.y - pos.y, p1.x - pos.x);

			float diff = to - from;

			if (diff < -M_PI) {
				diff += 2 * M_PI;
			}
			else if (diff > M_PI) {
				diff -= 2 * M_PI;
			}

			float fi = from + (diff * (i - 40) / 20);

			vertices[i] = vec2(pos.x + pos.z * cosf(fi), pos.y + pos.z * sinf(fi));
		}

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * 61,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL);
	}

	void Draw() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_LOOP, 0 /*startIdx*/, 60 /*# Elements*/);
	}
};

struct TriangleInside
{
	unsigned int vao, vbo;
	vec3 color;
	std::vector<vec2> oVertices;
	std::vector<vec2> drawVertices;
	
	TriangleInside(vec2 vertices[], vec3 color)
	{
		this->color = color;
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active
		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		for (size_t i = 0; i < 60; i++)
		{
			oVertices.push_back(vertices[i]);
		}

		while (oVertices.size() > 3) {
			for (size_t i = 0; i < oVertices.size(); i++)
			{
				int a = i - 1;
				int b = i;
				int c = i + 1;

				bool diagonal = true;
				bool inside = false;

				if (i == 0) a = oVertices.size() - 1;

				vec2 mid = vec2((oVertices[a].x + oVertices[c].x) / 2, (oVertices[a].y + oVertices[c].y) / 2);

				for (size_t j = 0; j < 60 - 1; j++)
				{
					if (intersect(oVertices[a], oVertices[c], vertices[j], vertices[j + 1])) {
						diagonal = false;
						break;
					}
				}

				if (intersect(oVertices[a], oVertices[c], vertices[59], vertices[0])) {
					diagonal = false;
				}

				if (diagonal) {
					int counter = 0;
					for (size_t k = 0; k < 60 - 1; k++)
					{
						if (intersect(mid, vec2(1.0f, 1.0f), vertices[k], vertices[k + 1])) {
							counter++;
						}
					}
						
					if (intersect(mid, vec2(1.0f, 1.0f), vertices[59], vertices[0])) counter++;
					if (counter % 2 == 1) inside = true;
					
				}

				if (diagonal && inside) {
					drawVertices.push_back(oVertices[a]);
					drawVertices.push_back(oVertices[b]);
					drawVertices.push_back(oVertices[c]);

					oVertices.erase(oVertices.begin() + b);
					break;
				}
			}
		}

		drawVertices.push_back(oVertices[0]);
		drawVertices.push_back(oVertices[1]);
		drawVertices.push_back(oVertices[2]);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * drawVertices.size(),  // # bytes
			&drawVertices[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	void Draw() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, drawVertices.size() /*# Elements*/);
	}
};

std::vector<Circle*> circles;
std::vector<TriangleSide*> sides;
TriangleInside* ti;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	
	circles.insert(circles.begin(), new Circle(vec2(0, 0), vec3(0.5f, 0.5f, 0.5f)));

	for (auto o : circles) {
		o->Draw();
	}

	if (ti) { ti->Draw(); }

	for (auto o : sides) {
		o->Draw();
	}

	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	
}


int counter = 0;
std::vector<vec2*> clicks;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char* buttonStat;
	switch (state) {
		case GLUT_DOWN: {
			counter++;
			buttonStat = "pressed";
			clicks.push_back(new vec2(cX, cY));

			circles.push_back(new Circle(vec2(cX, cY), vec3(1, 0, 0), 0.015f));
			if (counter >= 3) {

				sides.push_back(new TriangleSide(*clicks[0], *clicks[1], *clicks[2], vec3(1, 1, 0.5f)));
				ti = new TriangleInside(sides[0]->vertices, vec3(0, 0.05f, 1));

				float a1 = siriusAngle(*clicks[0], cPos[0], cPos[2], sides[0]->vertices[59], sides[0]->vertices[0], sides[0]->vertices[1]);
				float a2 = siriusAngle(*clicks[1], cPos[0], cPos[1], sides[0]->vertices[19], sides[0]->vertices[20], sides[0]->vertices[21]);
				float a3 = siriusAngle(*clicks[2], cPos[1], cPos[2], sides[0]->vertices[39], sides[0]->vertices[40], sides[0]->vertices[41]);
				
				printf("Alpha: %f, ", a3);
				printf("Beta %f, ", a2);
				printf("Gamma: %f, ", a1);


				printf("Angle sum: %f,\n", a3 + a2 + a1);

				float al = 0;
				float bl = 0;
				float cl = 0;

				for (size_t i = 0; i < 20; i++)
				{
					al += siriusLength(sides[0]->vertices[i], sides[0]->vertices[i+1]);
				}

				for (size_t i = 20; i < 40; i++)
				{
					bl += siriusLength(sides[0]->vertices[i], sides[0]->vertices[i + 1]);
				}

				for (size_t i = 40; i < 60-1; i++)
				{
					cl += siriusLength(sides[0]->vertices[i], sides[0]->vertices[i + 1]);
				}

				cl += siriusLength(sides[0]->vertices[59], sides[0]->vertices[0]);

				printf("a: %f, ", bl);
				printf("b: %f, ", al);
				printf("c: %f", cl);
			}

		}

	 glutSwapBuffers(); // exchange buffers for double buffering
	}

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
