using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class implicit_model : MonoBehaviour
{
	float 		t 		= 0.0333f;
	float 		mass	= 1;
	float		damping	= 0.99f;
	float 		rho		= 0.995f;
	float 		spring_k = 8000; // 弹力系数 k
	int[] 		E;
	float[] 	L;
	Vector3[] 	V;
	Vector3 Gravity = new Vector3(0.0f, -9.8f, 0.0f); // 重力加速度
	float r = 2.7f; // 碰撞球体半径，硬编码等于2.7


    void Start()
    {
		Mesh mesh = GetComponent<MeshFilter> ().mesh;

		// 生成布料数据
		int n=21;
		Vector3[] X  	= new Vector3[n*n];
		Vector2[] UV 	= new Vector2[n*n];
		int[] triangles	= new int[(n-1)*(n-1)*6];
		for(int j=0; j<n; j++)
		{
			for(int i=0; i<n; i++)
			{
				X[j*n+i] =new Vector3(5-10.0f*i/(n-1), 0, 5-10.0f*j/(n-1));
				UV[j*n+i]=new Vector3(i/(n-1.0f), j/(n-1.0f));
			}
		}
		int t=0;
		for(int j=0; j<n-1; j++)
		{
			for(int i=0; i<n-1; i++)	
			{
				triangles[t*6+0]=j*n+i;
				triangles[t*6+1]=j*n+i+1;
				triangles[t*6+2]=(j+1)*n+i+1;
				triangles[t*6+3]=j*n+i;
				triangles[t*6+4]=(j+1)*n+i+1;
				triangles[t*6+5]=(j+1)*n+i;
				t++;
			}
		}
		mesh.vertices=X;
		mesh.triangles=triangles;
		mesh.uv = UV;
		mesh.RecalculateNormals ();


		// 获取三角形边 E01, E12, E20
		int[] _E = new int[triangles.Length*2];
		for (int i=0; i<triangles.Length; i+=3) 
		{
			_E[i*2+0]=triangles[i+0];
			_E[i*2+1]=triangles[i+1];
			_E[i*2+2]=triangles[i+1];
			_E[i*2+3]=triangles[i+2];
			_E[i*2+4]=triangles[i+2];
			_E[i*2+5]=triangles[i+0];
		}
		// 交换边的顶点索引，索引小的在前
		for (int i=0; i<_E.Length; i+=2)
		{
			if(_E[i] > _E[i + 1]) 
				Swap(ref _E[i], ref _E[i+1]);
		}
		// 排序边，排序准则：顶点索引小排在前
		Quick_Sort (ref _E, 0, _E.Length/2-1);

		int e_number = 0; // 有效边数量，指剔除重复出现的边
		for (int i=0; i<_E.Length; i+=2)
		{
			if (i == 0 || _E [i + 0] != _E [i - 2] || _E [i + 1] != _E [i - 1]) 
				e_number++;
		}
		E = new int[e_number * 2]; // 有效边的集合
		for (int i=0, e=0; i<_E.Length; i+=2)
		{
			if (i == 0 || _E [i + 0] != _E [i - 2] || _E [i + 1] != _E [i - 1]) 
			{
				E[e*2+0]=_E [i + 0];
				E[e*2+1]=_E [i + 1];
				e++;
			}
		}

		L = new float[E.Length/2]; // 边的初始长度
		for (int e=0; e<E.Length/2; e++) 
		{
			int v0 = E[e*2+0];
			int v1 = E[e*2+1];
			L[e]=(X[v0]-X[v1]).magnitude;
		}

		V = new Vector3[X.Length]; // 顶点初始速度
		for (int i=0; i<V.Length; i++)
		{
			V[i] = new Vector3 (0, 0, 0);
		}
    }

    void Quick_Sort(ref int[] a, int l, int r)
	{
		int j;
		if(l<r)
		{
			j=Quick_Sort_Partition(ref a, l, r);
			Quick_Sort (ref a, l, j-1);
			Quick_Sort (ref a, j+1, r);
		}
	}

	int  Quick_Sort_Partition(ref int[] a, int l, int r)
	{
		int pivot_0, pivot_1, i, j;
		pivot_0 = a [l * 2 + 0];
		pivot_1 = a [l * 2 + 1];
		i = l;
		j = r + 1;
		while (true) 
		{
			do ++i; while( i<=r && (a[i*2]<pivot_0 || a[i*2]==pivot_0 && a[i*2+1]<=pivot_1));
			do --j; while(  a[j*2]>pivot_0 || a[j*2]==pivot_0 && a[j*2+1]> pivot_1);
			if(i>=j)	break;
			Swap(ref a[i*2], ref a[j*2]);
			Swap(ref a[i*2+1], ref a[j*2+1]);
		}
		Swap (ref a [l * 2 + 0], ref a [j * 2 + 0]);
		Swap (ref a [l * 2 + 1], ref a [j * 2 + 1]);
		return j;
	}

	void Swap(ref int a, ref int b)
	{
		int temp = a;
		a = b;
		b = temp;
	}

	void Collision_Handling()
	{
		Mesh mesh = GetComponent<MeshFilter> ().mesh;
		Vector3[] X = mesh.vertices;

		// 球体与布料碰撞处理
		GameObject sphere = GameObject.Find("Sphere");
		Vector3 sphereCenter = sphere.GetComponent<Transform>().position;

		for (int i = 0; i < X.Length; i ++)
		{
			float dist = (X[i] - sphereCenter).magnitude;
			if (dist < r) // 发生碰撞
			{
				X[i] = sphereCenter + r * (X[i] - sphereCenter) / dist; // 推到碰撞球体的表面
				V[i] += (sphereCenter + r * (X[i] - sphereCenter) / dist - X[i]) / t;
			}
		}

		mesh.vertices = X;
	}

	void Get_Gradient(Vector3[] X, Vector3[] X_hat, float t, Vector3[] G)
	{
		// 对每个顶点计算梯度，添加重力进来一块计算
		for (int i = 0; i < X.Length; i ++)
		{
			/* 
				一阶导
				g = M(x - x_hat) / ∆t - f(x), f(x) = mg 重力
			*/
			G[i] = mass * (X[i] - X_hat[i]) / (t * t) - mass * Gravity;
		}
		
		// 计算弹簧弹力，并加到对应顶点
		for (int e = 0; e < E.Length / 2; e ++)
		{
			int i = E[2 * e + 0];
			int j = E[2 * e + 1];
			float Le = L[e];
			float Lij = (X[i] - X[j]).magnitude;

			/* 
				Edge = {i, j}的弹簧弹力
				f_k(xi) = k (1 - Le / ||xi - xj||) (xi - xj), Le为原始弹簧长度
				f_k(xj) = - k (1 - Le / ||xi - xj||) (xi - xj), 反作用力
				弹力加入到一阶导的f(x)中
				gi = gi + f_k(xi), gj = gj + f_k(xj)
			*/
			G[i] += spring_k * (1 - Le / Lij) * (X[i] - X[j]);
			G[j] -= spring_k * (1 - Le / Lij) * (X[i] - X[j]);
		}
		
	}

    
	void Update () 
	{
		Mesh mesh = GetComponent<MeshFilter> ().mesh;
		Vector3[] X 		= mesh.vertices;
		Vector3[] last_X 	= new Vector3[X.Length];
		Vector3[] X_hat 	= new Vector3[X.Length];
		Vector3[] G 		= new Vector3[X.Length];

		/*
			implicit method
			核心要义：用下一刻的力来更新当前时刻的位置X和速度V
			作用部分：一阶导中的f(x)外力计算部分
			GAMES103 Lec5 page18
		*/

		// 初始化设置
		for (int i = 0; i < X.Length; i ++)
		{
			if (i == 0 || i == 20) // 布头两角固定住
				continue;
			last_X[i] = X[i];
			V[i] *= damping;
			X_hat[i] = X[i] + t * V[i];
			X[i] = X_hat[i];
		}

		// 简化版牛顿法，自身 - 一阶导/二阶导
		for(int k=0; k<32; k++)
		{
			Get_Gradient(X, X_hat, t, G);
			
			for (int i = 0; i < X.Length; i ++)
			{
				if (i == 0 || i == 20)
					continue;
				/* 
					简化后的二阶导，原版的Hessian太复杂
					g' = M / ∆t^2 + 4k
				*/
				X[i] -= G[i] / (mass / (t * t) + 4 * spring_k);
			}
		}

		// 更新位置和速度
		for (int i = 0; i < V.Length; i ++)
		{
			V[i] += (X[i] - X_hat[i]) / t;
		}
		mesh.vertices = X;

		// 处理碰撞
		Collision_Handling ();
		mesh.RecalculateNormals ();
	}
}
