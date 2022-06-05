using UnityEngine;
using System.Collections;

public class PBD_model: MonoBehaviour {

	float 		t= 0.0333f;
	float		damping= 0.99f;
	int[] 		E;
	float[] 	L;
	Vector3[] 	V;
	Vector3 gravity = new Vector3(0.0f, -9.8f, 0.0f);
	float alpha = 0.2f;
	float r = 2.7f; // 碰撞球体半径，硬编码等于2.7


	void Start () 
	{
		Mesh mesh = GetComponent<MeshFilter> ().mesh;

		int n=21;
		Vector3[] X  	= new Vector3[n*n];
		Vector2[] UV 	= new Vector2[n*n];
		int[] T	= new int[(n-1)*(n-1)*6];
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
				T[t*6+0]=j*n+i;
				T[t*6+1]=j*n+i+1;
				T[t*6+2]=(j+1)*n+i+1;
				T[t*6+3]=j*n+i;
				T[t*6+4]=(j+1)*n+i+1;
				T[t*6+5]=(j+1)*n+i;
				t++;
			}
		}
		mesh.vertices	= X;
		mesh.triangles	= T;
		mesh.uv 		= UV;
		mesh.RecalculateNormals ();


		int[] _E = new int[T.Length*2];
		for (int i=0; i<T.Length; i+=3) 
		{
			_E[i*2+0]=T[i+0];
			_E[i*2+1]=T[i+1];
			_E[i*2+2]=T[i+1];
			_E[i*2+3]=T[i+2];
			_E[i*2+4]=T[i+2];
			_E[i*2+5]=T[i+0];
		}
		for (int i=0; i<_E.Length; i+=2)
		{
			if(_E[i] > _E[i + 1]) 
				Swap(ref _E[i], ref _E[i+1]);
		}
		Quick_Sort (ref _E, 0, _E.Length/2-1);

		int e_number = 0;
		for (int i=0; i<_E.Length; i+=2)
		{
			if (i == 0 || _E [i + 0] != _E [i - 2] || _E [i + 1] != _E [i - 1]) 
				e_number++;
		}

		E = new int[e_number * 2];
		for (int i=0, e=0; i<_E.Length; i+=2)
		{
			if (i == 0 || _E [i + 0] != _E [i - 2] || _E [i + 1] != _E [i - 1]) 
			{
				E[e*2+0]=_E [i + 0];
				E[e*2+1]=_E [i + 1];
				e++;
			}
		}

		L = new float[E.Length/2];
		for (int e=0; e<E.Length/2; e++) 
		{
			int i = E[e*2+0];
			int j = E[e*2+1];
			L[e]=(X[i]-X[j]).magnitude;
		}

		V = new Vector3[X.Length];
		for (int i=0; i<X.Length; i++)
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

	void Strain_Limiting()
	{
		Mesh mesh = GetComponent<MeshFilter> ().mesh;
		Vector3[] vertices = mesh.vertices;

		/*
			PBD - Position Based Dynamic
			Why  - 显式 & 隐式积分求解在stiffness大的情况表现出不稳定 & 收敛慢问题。
			      采用约束解决上述问题。
			What - 弹簧的约束 - 每根弹簧尽可能恢复原长
			       Φ(θ) = |xi - xj| - L = 0
			How  - 1. 先各自更新速度和位置
			       2. 投影(以最小代价满足约束) - 计算满足约束的最近位置
				   3. 再次更新位置和速度
				   xi_new = xi - (mi / (mi + mj)) (|xi - xj| - Le) (xi - xj) / |xi - xj|  -- 详见 GAMES103 Lec6 page10
				   xj_new = xj + (mi / (mi + mj)) (|xi - xj| - Le) (xi - xj) / |xi - xj|
				   
			
			Strain Limiting
			Why  - PBD没有物理含义，整体表现收到迭代次数和顶点个数限制，迭代次数太多导致Locking.
			       作为PBD补充，保证PBD的模拟稳定。
			What - 1. PBD的步骤1采用隐式积分等方法更新位置
			       2. 放宽约束条件 - |xi - xj|靠近原长即可 - σ_min < |xi - xj| / L < σ_max
			How  - 1. 先各自更新速度和位置(隐式积分)
			       2. 投影 - 放宽约束计算满足约束的最近位置
				   3. 再次更新位置和速度
				   xi_new = xi - (mi / (mi + mj)) (|xi - xj| - σ0Le) (xi - xj) / |xi - xj|  -- 详见 GAMES103 Lec6 page16
				   xj_new = xj + (mi / (mi + mj)) (|xi - xj| - σ0Le) (xi - xj) / |xi - xj|
		*/
		Vector3[] sum_x = new Vector3[vertices.Length];
		float[] sum_n = new float[vertices.Length];

		for (int i = 0; i < vertices.Length; i ++)
		{
			sum_x[i] = new Vector3(0.0f, 0.0f, 0.0f);
			sum_n[i] = 0.0f;
		}
		for (int e = 0; e < E.Length / 2; e ++)
		{
			int i = E[e * 2 + 0];
			int j = E[e * 2 + 1];
			float Le = L[e];
			float Leij = (vertices[i] - vertices[j]).magnitude;

			sum_x[i] += vertices[i] - 0.5f * (Leij - Le) * (vertices[i] - vertices[j]) / Leij;
			sum_x[j] += vertices[j] + 0.5f * (Leij - Le) * (vertices[i] - vertices[j]) / Leij;

			sum_n[i] += 1;
			sum_n[j] += 1;
		}
		for (int i = 0; i < vertices.Length; i ++)
		{
			if (i == 0 || i == 20)
				continue;
			Vector3 xi = (sum_x[i] + alpha * vertices[i]) / (sum_n[i] + alpha);
			V[i] += (xi - vertices[i]) / t;
			vertices[i] = xi;
		}

		mesh.vertices = vertices;
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


	void Update () 
	{
		Mesh mesh = GetComponent<MeshFilter> ().mesh;
		Vector3[] X = mesh.vertices;

		for(int i=0; i<X.Length; i++)
		{
			if(i==0 || i==20)	
				continue;
			// 初始化更新速度v和位置x, 考虑重力影响
			V[i] *= damping;
			V[i] += gravity * t;
			X[i] += V[i] * t;
		}
		mesh.vertices = X;

		for(int l=0; l<32; l++)
		{
			// strian约束PBD - Position Based Dynamic
			Strain_Limiting ();
		}

		Collision_Handling ();

		mesh.RecalculateNormals ();

	}


}

