using UnityEngine;
using System.Collections;
using System;

public class Rigid_Bunny : MonoBehaviour 
{
	bool launched 		= false;
	float dt 			= 0.015f; // 步长
	Vector3 v 			= new Vector3(0, 0, 0);	// velocity 线速度
	Vector3 w 			= new Vector3(0, 0, 0);	// angular velocity 角速度
	
	float mass;									// mass 质量
	Matrix4x4 I_ref;							// reference inertia 惯性张量

	float linear_decay	= 0.999f;				// for velocity decay
	float angular_decay	= 0.98f;				
	float restitution 	= 0.5f;					// for collision 弹性系数
	float friction = 0.2f;						// 摩擦系数

	Vector3 gravity = new Vector3(0.0f, -9.8f, 0.0f); // 重力加速度


	// Use this for initialization
	void Start () 
	{		
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] vertices = mesh.vertices;

		float m=1; // 每个质点质量为1
		mass=0;
		for (int i=0; i<vertices.Length; i++) 
		{
			mass += m;
			float diag=m*vertices[i].sqrMagnitude;
			I_ref[0, 0]+=diag;
			I_ref[1, 1]+=diag;
			I_ref[2, 2]+=diag;
			I_ref[0, 0]-=m*vertices[i][0]*vertices[i][0];
			I_ref[0, 1]-=m*vertices[i][0]*vertices[i][1];
			I_ref[0, 2]-=m*vertices[i][0]*vertices[i][2];
			I_ref[1, 0]-=m*vertices[i][1]*vertices[i][0];
			I_ref[1, 1]-=m*vertices[i][1]*vertices[i][1];
			I_ref[1, 2]-=m*vertices[i][1]*vertices[i][2];
			I_ref[2, 0]-=m*vertices[i][2]*vertices[i][0];
			I_ref[2, 1]-=m*vertices[i][2]*vertices[i][1];
			I_ref[2, 2]-=m*vertices[i][2]*vertices[i][2];
		}
		I_ref [3, 3] = 1;
	}
	
	Matrix4x4 Get_Cross_Matrix(Vector3 a)
	{
		//Get the cross product matrix of vector a
		Matrix4x4 A = Matrix4x4.zero;
		A [0, 0] = 0; 
		A [0, 1] = -a [2]; 
		A [0, 2] = a [1]; 
		A [1, 0] = a [2]; 
		A [1, 1] = 0; 
		A [1, 2] = -a [0]; 
		A [2, 0] = -a [1]; 
		A [2, 1] = a [0]; 
		A [2, 2] = 0; 
		A [3, 3] = 1;
		return A;
	}

	Quaternion Add(Quaternion q1, Quaternion q2)
	{
		return new Quaternion(q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.w + q2.w);
	}

	Quaternion Sub(Quaternion q1, Quaternion q2)
	{
		return new Quaternion(q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.w - q2.w);
	}

	Matrix4x4 MatrixMultiplyFloat(Matrix4x4 mat, float coeff)
	{
		for (int i=0; i<4; i++)
		{
			for (int j=0; j<4; j++)
			{
				mat[i, j] *= coeff;
			}
		}
		return mat;
	}

	Matrix4x4 MatrixSub(Matrix4x4 mat1, Matrix4x4 mat2)
	{
		for (int i=0; i<4; i++)
		{
			for (int j=0; j<4; j++)
			{
				mat1[i, j] -= mat2[i, j];
			}
		}
		return mat1;
	}

	// In this function, update v and w by the impulse due to the collision with
	//a plane <P, N>
	void Collision_Impulse(Vector3 P, Vector3 N)
	{
		/*
			碰撞检测处理
			P -- 平面上一点
			N -- 平面法向量
			目标：更新 v & w
		*/

		// 1. 获取每个顶点局部坐标
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] vertices = mesh.vertices;

		// 2. 获取每个顶点的全局坐标系下的旋转R和平移T
		Matrix4x4 R = Matrix4x4.Rotate(transform.rotation);
		Vector3 T = transform.position;

		// 3. 计算平均碰撞点
		Vector3 sum = new Vector3(0, 0, 0); // 平均碰撞点
		int collisionNum = 0; // 碰撞点数目

		for (int i = 0; i < vertices.Length; i ++)
		{
			// 3.1 计算每个顶点到平面的距离
			Vector3 r_i = vertices[i]; // 每个顶点到原点的力矩
			Vector3 Rri = R.MultiplyVector(r_i);
			Vector3 x_i = T + Rri; // 顶点世界坐标
			float d = Vector3.Dot(x_i - P, N); // 到平面距离

			if (d < 0.0f) // 发生碰撞
			{
				// 3.2 判断物体是否还在向墙内运动
				Vector3 v_i = v + Vector3.Cross(w, Rri); // 碰撞点速度
				float vi_dot_N = Vector3.Dot(v_i, N); // 沿着碰撞平面法向方向速度
				
				if (vi_dot_N < 0.0f) // 小于零表示还有沿着法向向墙内速度
				{
					sum += r_i;
					collisionNum ++;
				}
			}
		}

		if (collisionNum == 0)
			return;
		
		// 3.3 更新平均碰撞点， 碰撞速度
		Vector3 r_collision = sum / collisionNum; // 平均碰撞点(局部坐标)
		Vector3 Rr_collision = R.MultiplyVector(r_collision);
		Vector3 v_collision = v + Vector3.Cross(w, Rr_collision); // 平均碰撞点的速度(世界坐标)

		// 4. 计算碰撞后的新速度
		Vector3 v_N = Vector3.Dot(v_collision, N) * N; // 法向分量
		Vector3 v_T = v_collision - v_N; // 切向分量
		Vector3 v_N_new = -1.0f * restitution * v_N;
		float a = Math.Max(1.0f - friction * (1.0f + restitution) * v_N.magnitude / v_T.magnitude, 0.0f);
		Vector3 v_T_new = a * v_T;
		Vector3 v_new = v_N_new + v_T_new; // 新速度

		// 5. 计算冲量j
		// 5.1 计算 转动惯量
		Matrix4x4 I_rot = R * I_ref * Matrix4x4.Transpose(R); // 转动惯量(全局)
		Matrix4x4 I_inv = Matrix4x4.Inverse(I_rot);
		// 5.2 计算K->j
		Matrix4x4 Rri_star = Get_Cross_Matrix(Rr_collision);
		Matrix4x4 K = MatrixSub(MatrixMultiplyFloat(Matrix4x4.identity, 1.0f / mass), Rri_star * I_inv * Rri_star);
		Vector3 J = K.inverse.MultiplyVector(v_new - v_collision);

		// 6. 更新 v 和 w
		v = v + 1.0f / mass * J;
		w = w + I_inv.MultiplyVector(Vector3.Cross(Rr_collision, J));
	}

	// Update is called once per frame
	void Update () 
	{
		//Game Control
		if(Input.GetKey("r"))
		{
			transform.position = new Vector3 (0, 0.6f, 0);
			restitution = 0.5f;
			friction = 0.2f;
			launched=false;
		}
		if(Input.GetKey("l"))
		{
			v = new Vector3 (5, 2, 0);
			launched=true;
		}

		if (launched)
		{
			// Part I: Update velocities
			v = v + dt * gravity; // 速度更新，v = v0 + gt
			v = v * linear_decay; // 速度衰减
			w = w * angular_decay; // 角速度衰减

			// Part II: Collision Impulse
			Collision_Impulse(new Vector3(0, 0.01f, 0), new Vector3(0, 1, 0)); // 与地面碰撞
			Collision_Impulse(new Vector3(2, 0, 0), new Vector3(-1, 0, 0)); // 与墙面碰撞

			// Part III: Update position & orientation
			//Update linear status
			Vector3 x    = transform.position;
			//Update angular status
			Quaternion q = transform.rotation;

			x = x + dt * v;
			Vector3 w_ = 0.5f * dt * w;
			Quaternion dw = new Quaternion(w_.x, w_.y, w_.z, 0.0f);
			q = Add(q, dw * q);
		
			// Part IV: Assign to the object
			transform.position = x;
			transform.rotation = q;
		}
	}
}
