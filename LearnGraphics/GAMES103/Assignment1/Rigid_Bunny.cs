using UnityEngine;
using System.Collections;
using System;

public class Rigid_Bunny : MonoBehaviour 
{
	bool launched 		= false;
	float dt 			= 0.015f;
	Vector3 v 			= new Vector3(0, 0, 0);	// velocity
	Vector3 w 			= new Vector3(0, 0, 0);	// angular velocity
	
	float mass;									// mass
	Matrix4x4 I_ref;							// reference inertia

	float linear_decay	= 0.999f;				// for velocity decay
	float angular_decay	= 0.98f;				
	float restitution 	= 0.5f;					// for collision
	float friction = 0.2f;

	Vector3 gravity = new Vector3(0.0f, -9.8f, 0.0f);


	// Use this for initialization
	void Start () 
	{		
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] vertices = mesh.vertices;

		float m=1;
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

	Quaternion QuaAdd(Quaternion q1, Quaternion q2)
	{
		return new Quaternion(q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.w + q2.w);
	}

	Matrix4x4 Mat4x4MulFloat(Matrix4x4 mat, float coeff)
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

	Matrix4x4 Mat4x4Sub(Matrix4x4 mat1, Matrix4x4 mat2)
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
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] vertices = mesh.vertices;

		// 获取全局旋转与平移
		Matrix4x4 R = Matrix4x4.Rotate(transform.rotation);
		Vector3 T = transform.position;

		// 找平均碰撞点
		Vector3 aveCollison = new Vector3(0.0f, 0.0f, 0.0f);
		int collisionCount = 0;

		for (int i = 0; i < vertices.Length; i ++)
		{
			Vector3 ri = vertices[i];
			Vector3 Rri = R.MultiplyVector(ri);
			Vector3 xi = Rri + T; // 全局坐标

			float dist = Vector3.Dot((xi - P), N); // 距离平面

			if (dist < 0.0f)
			{
				Vector3 vi = v + Vector3.Cross(w, Rri);
				if (Vector3.Dot(vi, N) < 0.0f) // 如果沿着平面法向反向有分量 -- 往墙内有速度
				{
					collisionCount ++;
					aveCollison += ri;
				}
			}
		}

		if (collisionCount == 0) // 无碰撞，直接返回
			return ;

		aveCollison /= collisionCount;
		Vector3 Rr_ave = R.MultiplyVector(aveCollison);
		Vector3 v_ave = v + Vector3.Cross(w, Rr_ave); // 平均碰撞点线速度

		if (Vector3.Dot(v_ave, N) < 0.0f) // 往墙内有速度
		{
			Vector3 v_N = Vector3.Dot(v_ave, N) * N; // 往墙内的速度 -- 法向速度
			Vector3 v_T = v_ave - v_N; // 切向

			Vector3 v_N_new = - restitution * v_N; // 法向分量乘上弹力系数
			float a = Math.Max(1 - friction * (1 + restitution) * v_N.magnitude / v_T.magnitude, 0.0f);
			Vector3 v_T_new = a * v_T; // 切向分量乘上摩擦系数

			Vector3 v_new = v_N_new + v_T_new; // 纠正后的碰撞后的速度

			Matrix4x4 I_rot = R * I_ref * R.transpose;
			Matrix4x4 I_inv = I_rot.inverse;

			Matrix4x4 Rri_star = Get_Cross_Matrix(Rr_ave);
			Matrix4x4 K = Mat4x4Sub(Mat4x4MulFloat(Matrix4x4.identity, 1.0f / mass), Rri_star * I_inv * Rri_star);

			Vector3 J = K.inverse.MultiplyVector(v_new - v_ave); // 冲量

			// 更新 线速度 v + 角速度 w
			v = v + J / mass;
			w = w + I_inv.MultiplyVector(Vector3.Cross(Rr_ave, J));
		}
	}

	// Update is called once per frame
	void Update () 
	{
		//Game Control
		if(Input.GetKey("r"))
		{
			transform.position = new Vector3 (0, 0.6f, 0);
			restitution = 0.5f;
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
			v += dt * gravity;
			v *= linear_decay;
			w *= angular_decay;

			// Part II: Collision Impulse
			Collision_Impulse(new Vector3(0, 0.01f, 0), new Vector3(0, 1, 0));
			Collision_Impulse(new Vector3(2, 0, 0), new Vector3(-1, 0, 0));

			// Part III: Update position & orientation
			//Update linear status
			Vector3 x    = transform.position;
			//Update angular status
			Quaternion q = transform.rotation;

			x += v * dt; // 位置更新
			Vector3 w_ = 0.5f * dt * w;
			Quaternion dw = new Quaternion(w_.x, w_.y, w_.z, 0.0f);
			q = QuaAdd(q, dw * q); // 方向更新

			// Part IV: Assign to the object
			transform.position = x;
			transform.rotation = q;
		}
	}
}
