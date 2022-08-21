using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class FVM : MonoBehaviour
{
	float dt 			= 0.003f;
    float mass 			= 1;
	float stiffness_0	= 20000.0f; // lambda
    float stiffness_1 	= 5000.0f; // mu
    float damp			= 0.999f;

	int[] 		Tet;
	int tet_number;			//The number of tetrahedra

	Vector3[] 	Force;
	Vector3[] 	V;
	Vector3[] 	X;
	int number;				//The number of vertices

	Matrix4x4[] inv_Dm;

	//For Laplacian smoothing.
	Vector3[]   V_sum;
	int[]		V_num;

	SVD svd = new SVD();

	// gravity
	Vector3 gravity = new Vector3(0.0f, -9.8f, 0.0f);

	// collision
	Vector3 floor;
	Vector3 floor_N;
	float restitution 	= 0.5f;					// for collision
	float friction = 0.2f;

    // Start is called before the first frame update
    void Start()
    {
    	// FILO IO: Read the house model from files.
    	// The model is from Jonathan Schewchuk's Stellar lib.
    	{
    		string fileContent = File.ReadAllText("Assets/house2.ele");
    		string[] Strings = fileContent.Split(new char[]{' ', '\t', '\r', '\n'}, StringSplitOptions.RemoveEmptyEntries);
    		
    		tet_number=int.Parse(Strings[0]);
        	Tet = new int[tet_number*4];

    		for(int tet=0; tet<tet_number; tet++)
    		{
				Tet[tet*4+0]=int.Parse(Strings[tet*5+4])-1;
				Tet[tet*4+1]=int.Parse(Strings[tet*5+5])-1;
				Tet[tet*4+2]=int.Parse(Strings[tet*5+6])-1;
				Tet[tet*4+3]=int.Parse(Strings[tet*5+7])-1;
			}
    	}
    	{
			string fileContent = File.ReadAllText("Assets/house2.node");
    		string[] Strings = fileContent.Split(new char[]{' ', '\t', '\r', '\n'}, StringSplitOptions.RemoveEmptyEntries);
    		number = int.Parse(Strings[0]);
    		X = new Vector3[number];
       		for(int i=0; i<number; i++)
       		{
       			X[i].x=float.Parse(Strings[i*5+5])*0.4f;
       			X[i].y=float.Parse(Strings[i*5+6])*0.4f;
       			X[i].z=float.Parse(Strings[i*5+7])*0.4f;
       		}
    		//Centralize the model.
	    	Vector3 center=Vector3.zero;
	    	for(int i=0; i<number; i++)		
				center+=X[i];
	    	center=center/number;
	    	
			for(int i=0; i<number; i++)
	    	{
	    		X[i]-=center;
	    		float temp=X[i].y;
	    		X[i].y=X[i].z;
	    		X[i].z=temp;
	    	}
		}
        /*tet_number=1;
        Tet = new int[tet_number*4];
        Tet[0]=0;
        Tet[1]=1;
        Tet[2]=2;
        Tet[3]=3;

        number=4;
        X = new Vector3[number];
        V = new Vector3[number];
        Force = new Vector3[number];
        X[0]= new Vector3(0, 0, 0);
        X[1]= new Vector3(1, 0, 0);
        X[2]= new Vector3(0, 1, 0);
        X[3]= new Vector3(0, 0, 1);*/


        //Create triangle mesh.
       	Vector3[] vertices = new Vector3[tet_number*12];
        int vertex_number=0;
        for(int tet=0; tet<tet_number; tet++)
        {
        	vertices[vertex_number++]=X[Tet[tet*4+0]];
        	vertices[vertex_number++]=X[Tet[tet*4+2]];
        	vertices[vertex_number++]=X[Tet[tet*4+1]];

        	vertices[vertex_number++]=X[Tet[tet*4+0]];
        	vertices[vertex_number++]=X[Tet[tet*4+3]];
        	vertices[vertex_number++]=X[Tet[tet*4+2]];

        	vertices[vertex_number++]=X[Tet[tet*4+0]];
        	vertices[vertex_number++]=X[Tet[tet*4+1]];
        	vertices[vertex_number++]=X[Tet[tet*4+3]];

        	vertices[vertex_number++]=X[Tet[tet*4+1]];
        	vertices[vertex_number++]=X[Tet[tet*4+2]];
        	vertices[vertex_number++]=X[Tet[tet*4+3]];
        }

        int[] triangles = new int[tet_number*12];
        for(int t=0; t<tet_number*4; t++)
        {
        	triangles[t*3+0]=t*3+0;
        	triangles[t*3+1]=t*3+1;
        	triangles[t*3+2]=t*3+2;
        }
        Mesh mesh = GetComponent<MeshFilter> ().mesh;
		mesh.vertices  = vertices;
		mesh.triangles = triangles;
		mesh.RecalculateNormals ();


		V 	  = new Vector3[number];
        Force = new Vector3[number];
        V_sum = new Vector3[number];
        V_num = new int[number];

		//TODO: Need to allocate and assign inv_Dm
		inv_Dm = new Matrix4x4[tet_number];
		for (int tet = 0; tet < tet_number; tet ++)
		{
			inv_Dm[tet] = Build_Edge_Matrix(tet).inverse;
		}

		// floor collision
		floor = GameObject.Find("Floor").transform.position;
		floor_N = new Vector3(0, 1, 0);
    }

    Matrix4x4 Build_Edge_Matrix(int tet)
    {
    	Matrix4x4 ret=Matrix4x4.zero;
    	//TODO: Need to build edge matrix here.
		Vector4 X10 = X[Tet[tet*4 + 1]] - X[Tet[tet*4 + 0]];
		Vector4 X20 = X[Tet[tet*4 + 2]] - X[Tet[tet*4 + 0]];
		Vector4 X30 = X[Tet[tet*4 + 3]] - X[Tet[tet*4 + 0]];

		ret.SetColumn(0, X10);
		ret.SetColumn(1, X20);
		ret.SetColumn(2, X30);
		ret[3, 3] = 1;

		return ret;
    }

	Matrix4x4 Matrix4x4Substract(Matrix4x4 m1, Matrix4x4 m2)
	{
		return new Matrix4x4(
			m1.GetColumn(0) - m2.GetColumn(0),
			m1.GetColumn(1) - m2.GetColumn(1),
			m1.GetColumn(2) - m2.GetColumn(2),
			m1.GetColumn(3) - m2.GetColumn(3)
		);
	}

	Matrix4x4 Matrix4x4Add(Matrix4x4 m1, Matrix4x4 m2)
	{
		return new Matrix4x4(
			m1.GetColumn(0) + m2.GetColumn(0),
			m1.GetColumn(1) + m2.GetColumn(1),
			m1.GetColumn(2) + m2.GetColumn(2),
			m1.GetColumn(3) + m2.GetColumn(3)
		);
	}

	Matrix4x4 Matrix4x4MulFloat(Matrix4x4 m, float num)
	{
		return new Matrix4x4(
			m.GetColumn(0) * num,
			m.GetColumn(1) * num,
			m.GetColumn(2) * num,
			m.GetColumn(3) * num
		);
	}

	float Matrix4x4Trace(Matrix4x4 m)
	{
		return m[0, 0] + m[1, 1] + m[2, 2];
	}


    void _Update()
    {
    	// Jump up.
		if(Input.GetKeyDown(KeyCode.Space))
    	{
    		for(int i=0; i<number; i++)
    			V[i].y+=0.2f;
    	}

    	for(int i=0; i<number; i++)
    	{
    		//TODO: Add gravity to Force.
			Force[i] = mass * gravity;

			// V_sum & V_num set zero
			V_sum[i] = new Vector3(0.0f, 0.0f, 0.0f);
			V_num[i] = 0;
    	}

    	for(int tet=0; tet<tet_number; tet++)
    	{
			Matrix4x4 Dm_inv = inv_Dm[tet];
    		//TODO: Deformation Gradient
			Matrix4x4 F = Build_Edge_Matrix(tet) * Dm_inv;
    		
    		//TODO: Green Strain
			Matrix4x4 G = Matrix4x4MulFloat(Matrix4x4Substract(F.transpose * F, Matrix4x4.identity), 0.5f);

    		//TODO: Second PK Stress
			Matrix4x4 S = Matrix4x4Add(Matrix4x4MulFloat(G, 2 * stiffness_1), Matrix4x4MulFloat(Matrix4x4.identity, stiffness_0 * Matrix4x4Trace(G)));

			// First PK Stress
			Matrix4x4 P = F * S;

    		//TODO: Elastic Force
			Matrix4x4 f = Matrix4x4MulFloat(P * Dm_inv.transpose, - 1.0f / 6.0f / Dm_inv.determinant);
			Vector3 f1 = f.GetColumn(0);
			Vector3 f2 = f.GetColumn(1);
			Vector3 f3 = f.GetColumn(2);
			Vector3 f0 = - f1 - f2 - f3;

			Force[Tet[tet*4 + 0]] += f0;
			Force[Tet[tet*4 + 1]] += f1;
			Force[Tet[tet*4 + 2]] += f2;
			Force[Tet[tet*4 + 3]] += f3;
    	}

    	for(int i=0; i<number; i++)
    	{
    		//TODO: Update X and V here.
			V[i] += Force[i] / mass * dt;
			V[i] *= damp;
    	}

		// for laplacian smoothing
		for (int tet = 0; tet < tet_number; tet ++)
		{
			int v_id0 = Tet[tet*4 + 0];
			int v_id1 = Tet[tet*4 + 1];
			int v_id2 = Tet[tet*4 + 2];
			int v_id3 = Tet[tet*4 + 3];

			Vector3 V0 = V[v_id0];
			Vector3 V1 = V[v_id1];
			Vector3 V2 = V[v_id2];
			Vector3 V3 = V[v_id3];

			V_sum[v_id0] += (V1 + V2 + V3) / 3;
			V_sum[v_id1] += (V0 + V2 + V3) / 3;
			V_sum[v_id2] += (V0 + V1 + V3) / 3;
			V_sum[v_id3] += (V0 + V1 + V2) / 3;

			V_num[v_id0] += 1;
			V_num[v_id1] += 1;
			V_num[v_id2] += 1;
			V_num[v_id3] += 1;
		}

		for (int i = 0; i < number; i ++)
		{
			// Laplacian Smoothing
			V[i] = (V_sum[i] / V_num[i] + V[i]) / 2;
			
			// Update X
			X[i] += V[i] * dt;

			// V_sum & V_num set zero
			V_sum[i] = new Vector3(0.0f, 0.0f, 0.0f);
			V_num[i] = 0;
		
			//TODO: (Particle) collision with floor.
			if (Vector3.Dot(X[i] - floor, floor_N) < 0.0f)
			{
				if (Vector3.Dot(V[i], floor_N) < 0.0f)
				{
					Vector3 v_N = Vector3.Dot(V[i], floor_N) * floor_N; // 往墙内的速度 -- 法向速度
					Vector3 v_T = V[i] - v_N; // 切向

					Vector3 v_N_new = - restitution * v_N; // 法向分量乘上弹力系数
					float a = Math.Max(1 - friction * (1 + restitution) * v_N.magnitude / v_T.magnitude, 0.0f);
					Vector3 v_T_new = a * v_T; // 切向分量乘上摩擦系数

					Vector3 v_new = v_N_new + v_T_new; // 纠正后的碰撞后的速度
					V[i] = v_new;
					X[i] = X[i] - Vector3.Dot(X[i] - floor, floor_N) * floor_N;
				}
			}
			Force[i] = new Vector3(0.0f, 0.0f, 0.0f);
		}

    }

    // Update is called once per frame
    void Update()
    {
    	for(int l=0; l<10; l++)
			_Update();

    	// Dump the vertex array for rendering.
    	Vector3[] vertices = new Vector3[tet_number*12];
        int vertex_number=0;
        for(int tet=0; tet<tet_number; tet++)
        {
        	vertices[vertex_number++]=X[Tet[tet*4+0]];
        	vertices[vertex_number++]=X[Tet[tet*4+2]];
        	vertices[vertex_number++]=X[Tet[tet*4+1]];
        	vertices[vertex_number++]=X[Tet[tet*4+0]];
        	vertices[vertex_number++]=X[Tet[tet*4+3]];
        	vertices[vertex_number++]=X[Tet[tet*4+2]];
        	vertices[vertex_number++]=X[Tet[tet*4+0]];
        	vertices[vertex_number++]=X[Tet[tet*4+1]];
        	vertices[vertex_number++]=X[Tet[tet*4+3]];
        	vertices[vertex_number++]=X[Tet[tet*4+1]];
        	vertices[vertex_number++]=X[Tet[tet*4+2]];
        	vertices[vertex_number++]=X[Tet[tet*4+3]];
        }
        Mesh mesh = GetComponent<MeshFilter> ().mesh;
		mesh.vertices  = vertices;
		mesh.RecalculateNormals ();
    }
}
